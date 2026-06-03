"""
主运行脚本
滑坡位移预测与联合预警 - 完整流程
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import pandas as pd
from config import Config

# 导入自定义模块
from src.data_loader import DataLoader
from src.preprocess import DataPreprocessor
from src.models import MultiTaskGRU, AttentionGRU
from src.train import Trainer
from src.evaluate import Evaluator
from src.utils import (
    set_seed, save_metrics, save_predictions,
    print_model_summary, check_gpu_availability,
    create_output_dirs
)


def run_baseline_model(config, train_loader, val_loader, test_loader,
                       device, feature_dim, preprocessor, test_data):
    """运行基线模型"""
    print("\n" + "=" * 60)
    print("第一部分：基线模型 (MultiTaskGRU)")
    print("=" * 60)

    # 创建模型
    baseline_model = MultiTaskGRU(
        input_size=feature_dim,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    )

    print_model_summary(baseline_model)

    # 训练
    trainer = Trainer(baseline_model, device)
    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=config.EPOCHS,
        lambda_risk=config.LAMBDA
    )

    # 保存模型
    trainer.save_model(f"{config.MODEL_SAVE_PATH}/baseline_model.pth")

    # 评估
    evaluator = Evaluator(baseline_model, device)
    metrics, disp_pred, disp_true, risk_pred, risk_true, risk_prob = \
        evaluator.evaluate(test_loader)

    print("\n基线模型测试集结果:")
    print("-" * 40)
    print(f"位移预测:")
    print(f"  h1 - MAE: {metrics['mae_h1']:.3f}, RMSE: {metrics['rmse_h1']:.3f}")
    print(f"  h2 - MAE: {metrics['mae_h2']:.3f}, RMSE: {metrics['rmse_h2']:.3f}")
    print(f"  h3 - MAE: {metrics['mae_h3']:.3f}, RMSE: {metrics['rmse_h3']:.3f}")
    print(f"  累计位移 - MAE: {metrics['mae_cum']:.3f}, RMSE: {metrics['rmse_cum']:.3f}, R²: {metrics['r2_cum']:.3f}")
    print(f"  最大位移 - MAE: {metrics['mae_max']:.3f}, RMSE: {metrics['rmse_max']:.3f}")
    print(f"  平均位移 - MAE: {metrics['mae_avg']:.3f}, RMSE: {metrics['rmse_avg']:.3f}")
    print(f"\n风险预测:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1-score: {metrics['f1']:.3f}")

    return baseline_model, metrics, disp_pred, disp_true, risk_pred, risk_true, risk_prob


def run_improved_model(config, train_loader, val_loader, test_loader,
                       device, feature_dim, preprocessor, test_data):
    """运行改进模型"""
    print("\n" + "=" * 60)
    print("第二部分：改进模型 (AttentionGRU)")
    print("=" * 60)

    # 创建模型
    improved_model = AttentionGRU(
        input_size=feature_dim,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    )

    print_model_summary(improved_model)

    # 训练
    trainer = Trainer(improved_model, device)
    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=config.EPOCHS,
        lambda_risk=config.LAMBDA
    )

    # 保存模型
    trainer.save_model(f"{config.MODEL_SAVE_PATH}/improved_model.pth")

    # 评估
    evaluator = Evaluator(improved_model, device)
    metrics, disp_pred, disp_true, risk_pred, risk_true, risk_prob = \
        evaluator.evaluate(test_loader)

    print("\n改进模型测试集结果:")
    print("-" * 40)
    print(f"位移预测:")
    print(f"  h1 - MAE: {metrics['mae_h1']:.3f}, RMSE: {metrics['rmse_h1']:.3f}")
    print(f"  h2 - MAE: {metrics['mae_h2']:.3f}, RMSE: {metrics['rmse_h2']:.3f}")
    print(f"  h3 - MAE: {metrics['mae_h3']:.3f}, RMSE: {metrics['rmse_h3']:.3f}")
    print(f"  累计位移 - MAE: {metrics['mae_cum']:.3f}, RMSE: {metrics['rmse_cum']:.3f}, R²: {metrics['r2_cum']:.3f}")
    print(f"  最大位移 - MAE: {metrics['mae_max']:.3f}, RMSE: {metrics['rmse_max']:.3f}")
    print(f"  平均位移 - MAE: {metrics['mae_avg']:.3f}, RMSE: {metrics['rmse_avg']:.3f}")
    print(f"\n风险预测:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1-score: {metrics['f1']:.3f}")

    return improved_model, metrics, disp_pred, disp_true, risk_pred, risk_true, risk_prob


def compare_models(baseline_metrics, improved_metrics, config):
    """对比基线和改进模型"""
    print("\n" + "=" * 60)
    print("模型对比分析")
    print("=" * 60)

    # 计算改进幅度
    print("\n位移预测改进幅度:")
    print("-" * 60)

    disp_metrics = ['mae_h1', 'mae_h2', 'mae_h3', 'mae_cum', 'rmse_cum', 'r2_cum']

    for metric in disp_metrics:
        baseline_val = baseline_metrics[metric]
        improved_val = improved_metrics[metric]

        if 'r2' in metric:
            # R²越大越好
            improvement = (improved_val - baseline_val) / abs(baseline_val) * 100 if baseline_val != 0 else 0
        else:
            # 其他指标越小越好
            improvement = (baseline_val - improved_val) / baseline_val * 100 if baseline_val != 0 else 0

        direction = "↑" if ('r2' in metric and improvement > 0) or ('r2' not in metric and improvement > 0) else "↓"
        print(f"  {metric}: {baseline_val:.4f} -> {improved_val:.4f} ({improvement:+.2f}% {direction})")

    print(f"\n风险预测改进幅度:")
    print("-" * 60)

    risk_metrics = ['accuracy', 'precision', 'recall', 'f1']

    for metric in risk_metrics:
        baseline_val = baseline_metrics[metric]
        improved_val = improved_metrics[metric]

        # 这些指标越大越好
        improvement = (improved_val - baseline_val) / baseline_val * 100 if baseline_val != 0 else 0
        direction = "↑" if improvement > 0 else "↓"
        print(f"  {metric}: {baseline_val:.4f} -> {improved_val:.4f} ({improvement:+.2f}% {direction})")


def main():
    """主函数"""
    print("=" * 60)
    print("滑坡位移预测与联合预警系统")
    print("=" * 60)

    # 初始化
    config = Config()
    set_seed(config.SEED)
    device = check_gpu_availability()
    create_output_dirs()

    # ============================================
    # 步骤1: 加载数据
    # ============================================
    print("\n" + "=" * 60)
    print("步骤1: 加载数据")
    print("=" * 60)

    data_loader = DataLoader()
    samples, features, nodes = data_loader.load_all_data()

    # ============================================
    # 步骤2: 数据预处理
    # ============================================
    print("\n" + "=" * 60)
    print("步骤2: 数据预处理")
    print("=" * 60)

    preprocessor = DataPreprocessor()

    # 划分数据集（按时间顺序）
    train_data, val_data, test_data = preprocessor.split_by_time(samples)

    # 准备特征和目标
    feature_cols, target_cols = preprocessor.prepare_features_targets(samples)
    feature_dim = len(feature_cols)

    # 提取特征和目标
    train_X, train_y_disp, train_y_risk = preprocessor.extract_features_targets(train_data)
    val_X, val_y_disp, val_y_risk = preprocessor.extract_features_targets(val_data)
    test_X, test_y_disp, test_y_risk = preprocessor.extract_features_targets(test_data)

    # 标准化（仅在训练集上计算统计量）
    train_X_scaled, val_X_scaled, test_X_scaled = preprocessor.normalize_features(
        train_X, val_X, test_X
    )

    # ============================================
    # 步骤3: 创建数据加载器
    # ============================================
    print("\n" + "=" * 60)
    print("步骤3: 创建数据加载器")
    print("=" * 60)

    # 注意：这里的batch处理方式需要调整
    # 由于数据已经是样本形式（不是序列），我们直接使用
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X_scaled),
        torch.FloatTensor(train_y_disp),
        torch.LongTensor(train_y_risk)
    )

    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_X_scaled),
        torch.FloatTensor(val_y_disp),
        torch.LongTensor(val_y_risk)
    )

    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X_scaled),
        torch.FloatTensor(test_y_disp),
        torch.LongTensor(test_y_risk)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )

    print(f"训练批次: {len(train_loader)}")
    print(f"验证批次: {len(val_loader)}")
    print(f"测试批次: {len(test_loader)}")

    # ============================================
    # 步骤4: 运行基线模型
    # ============================================
    baseline_model, baseline_metrics, baseline_disp_pred, baseline_disp_true, \
        baseline_risk_pred, baseline_risk_true, baseline_risk_prob = \
        run_baseline_model(config, train_loader, val_loader, test_loader,
                           device, feature_dim, preprocessor, test_data)

    # ============================================
    # 步骤5: 运行改进模型
    # ============================================
    improved_model, improved_metrics, improved_disp_pred, improved_disp_true, \
        improved_risk_pred, improved_risk_true, improved_risk_prob = \
        run_improved_model(config, train_loader, val_loader, test_loader,
                           device, feature_dim, preprocessor, test_data)

    # ============================================
    # 步骤6: 模型对比
    # ============================================
    compare_models(baseline_metrics, improved_metrics, config)

    # ============================================
    # 步骤7: 保存结果
    # ============================================
    print("\n" + "=" * 60)
    print("步骤7: 保存结果")
    print("=" * 60)

    # 保存评估指标
    all_metrics = {
        'baseline': baseline_metrics,
        'improved': improved_metrics
    }
    save_metrics(all_metrics, f"{config.RESULTS_PATH}/metrics.json")

    # 保存预测结果（使用改进模型的结果）
    save_predictions(
        test_data,
        improved_disp_pred, improved_disp_true,
        improved_risk_pred, improved_risk_true,
        improved_risk_prob,
        preprocessor.label_encoder,
        f"{config.RESULTS_PATH}/pred_test.csv"
    )

    # ============================================
    # 步骤8: 分区分析与可视化
    # ============================================
    print("\n" + "=" * 60)
    print("步骤8: 分区分析与可视化")
    print("=" * 60)

    # 使用改进模型进行评估
    evaluator = Evaluator(improved_model, device)

    # 分区分分析
    zone_metrics = evaluator.evaluate_by_zone(test_loader, test_data, nodes)

    # 误差分析
    errors, misclassified = evaluator.analyze_errors(
        test_loader, test_data, preprocessor.label_encoder
    )

    # 绘制预测曲线
    evaluator._plot_prediction_curve(test_data, config.FIGURES_PATH)

    # 绘制误差分布
    evaluator._plot_error_distribution(errors, config.FIGURES_PATH)

    # 绘制混淆矩阵
    evaluator._plot_confusion_matrix(
        improved_metrics['confusion_matrix'],
        preprocessor.label_encoder,
        config.FIGURES_PATH
    )

    # 绘制特征重要性
    evaluator._plot_feature_importance(config.FIGURES_PATH)

    # ============================================
    # 完成
    # ============================================
    print("\n" + "=" * 60)
    print("项目完成！")
    print("=" * 60)

    print(f"\n输出文件位置:")
    print(f"  模型文件: {config.MODEL_SAVE_PATH}/")
    print(f"  预测结果: {config.RESULTS_PATH}/pred_test.csv")
    print(f"  评估指标: {config.RESULTS_PATH}/metrics.json")
    print(f"  可视化图表: {config.FIGURES_PATH}/")

    print(f"\n主要结果摘要:")
    print(f"  位移预测 (累计) - 基线MAE: {baseline_metrics['mae_cum']:.3f}, 改进MAE: {improved_metrics['mae_cum']:.3f}")
    print(f"  风险预测 - 基线F1: {baseline_metrics['f1']:.3f}, 改进F1: {improved_metrics['f1']:.3f}")


if __name__ == "__main__":
    main()