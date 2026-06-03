import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os
from config import Config


class Trainer:
    """模型训练器"""

    def __init__(self, model, device=None):
        self.config = Config()
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=1e-5
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=10,
            factor=0.5,
            verbose=True
        )

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def create_dataloader(self, X, y_disp, y_risk, batch_size=32, shuffle=True):
        """创建数据加载器"""
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y_disp),
            torch.LongTensor(y_risk)
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        return dataloader

    def train_epoch(self, train_loader, lambda_risk=0.5):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_disp_loss = 0
        total_risk_loss = 0

        progress_bar = tqdm(train_loader, desc='Training')

        for X_batch, y_disp_batch, y_risk_batch in progress_bar:
            X_batch = X_batch.to(self.device)
            y_disp_batch = y_disp_batch.to(self.device)
            y_risk_batch = y_risk_batch.to(self.device)

            self.optimizer.zero_grad()

            disp_pred, risk_pred = self.model(X_batch)

            loss_disp = self.mse_loss(disp_pred, y_disp_batch)
            loss_risk = self.ce_loss(risk_pred, y_risk_batch)
            loss = loss_disp + lambda_risk * loss_risk

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            total_disp_loss += loss_disp.item()
            total_risk_loss += loss_risk.item()

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'disp': f'{loss_disp.item():.4f}',
                'risk': f'{loss_risk.item():.4f}'
            })

        n_batches = len(train_loader)
        return total_loss / n_batches, total_disp_loss / n_batches, total_risk_loss / n_batches

    def validate_epoch(self, val_loader, lambda_risk=0.5):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for X_batch, y_disp_batch, y_risk_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_disp_batch = y_disp_batch.to(self.device)
                y_risk_batch = y_risk_batch.to(self.device)

                disp_pred, risk_pred = self.model(X_batch)

                loss_disp = self.mse_loss(disp_pred, y_disp_batch)
                loss_risk = self.ce_loss(risk_pred, y_risk_batch)
                loss = loss_disp + lambda_risk * loss_risk

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader=None, epochs=None, lambda_risk=None):
        """完整训练流程"""
        if epochs is None:
            epochs = self.config.EPOCHS
        if lambda_risk is None:
            lambda_risk = self.config.LAMBDA

        print(f"\n开始训练...")
        print(f"设备: {self.device}")
        print(f"训练轮数: {epochs}")
        print(f"风险损失权重: {lambda_risk}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(epochs):
            # 训练
            train_loss, train_disp_loss, train_risk_loss = self.train_epoch(
                train_loader, lambda_risk
            )
            self.train_losses.append(train_loss)

            # 验证
            if val_loader is not None:
                val_loss = self.validate_epoch(val_loader, lambda_risk)
                self.val_losses.append(val_loss)

                # 学习率调度
                self.scheduler.step(val_loss)

                # 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()
                    print(f"  ✓ 新的最佳模型 (Val Loss: {val_loss:.4f})")

                print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"  Disp Loss: {train_disp_loss:.4f} | Risk Loss: {train_risk_loss:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f}")

        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n✓ 已加载最佳模型 (Val Loss: {self.best_val_loss:.4f})")

        return self.train_losses, self.val_losses

    def save_model(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, path)
        print(f"✓ 模型已保存到 {path}")

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"✓ 模型已加载自 {path}")