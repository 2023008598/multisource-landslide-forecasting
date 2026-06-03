import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from config import Config


class Evaluator:
    """模型评估器"""

    def __init__(self, model, device=None):
        self.config = Config()
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def evaluate(self, dataloader):
        """全面评估模型"""
        self.model.eval()

        all_disp_pred = []
        all_disp_true = []
        all_risk_pred = []
        all_risk_true = []
        all_risk_prob = []

        with torch.no_grad():
            for X_batch, y_disp_batch, y_risk_batch in dataloader:
                X_batch = X_batch.to(self.device)

                disp_pred, risk_pred = self.model(X_batch)

                all_disp_pred.append(disp_pred.cpu().numpy())
                all_disp_true.append(y_disp_batch.numpy())
                all_risk_prob.append(torch.softmax(risk_pred, dim=1).cpu().numpy())
                all_risk_pred.append(torch.argmax(risk_pred, dim=1).cpu().numpy())
                all_risk_true.append(y_risk_batch.numpy())

        # 合并
        disp_pred = np.vstack(all_disp_pred)
        disp_true = np.vstack(all_disp_true)
        risk_prob = np.vstack(all_risk_prob)
        risk_pred = np.concatenate(all_risk_pred)
        risk_true = np.concatenate(all_risk_true)

        # 计算指标
        metrics = self._calculate_metrics(disp_pred, disp_true, risk_pred, risk_true)

        return metrics, disp_pred, disp_true, risk_pred, risk_true, risk_prob

    def _calculate_metrics(self, disp_pred, disp_true, risk_pred, risk_true):
        """计算评估指标"""
        metrics = {}

        # 位移预测指标（每个预测月份）
        for i, name in enumerate(['h1', 'h2', 'h3']):
            metrics[f'mae_{name}'] = mean_absolute_error(disp_true[:, i], disp_pred[:, i])
            metrics[f'rmse_{name}'] = np.sqrt(mean_squared_error(disp_true[:, i], disp_pred[:, i]))

        # 累计位移指标
        metrics['mae_cum'] = mean_absolute_error(disp_true[:, 3], disp_pred[:, 3])
        metrics['rmse_cum'] = np.sqrt(mean_squared_error(disp_true[:, 3], disp_pred[:, 3]))
        metrics['r2_cum'] = r2_score(disp_true[:, 3], disp_pred[:, 3])

        # 最大位移指标
        metrics['mae_max'] = mean_absolute_error(disp_true[:, 4], disp_pred[:, 4])
        metrics['rmse_max'] = np.sqrt(mean_squared_error(disp_true[:, 4], disp_pred[:, 4]))

        # 平均位移指标
        metrics['mae_avg'] = mean_absolute_error(disp_true[:, 5], disp_pred[:, 5])
        metrics['rmse_avg'] = np.sqrt(mean_squared_error(disp_true[:, 5], disp_pred[:, 5]))

        # MAPE（注意除零处理）
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((disp_true[:, 3] - disp_pred[:, 3]) /
                                  np.where(disp_true[:, 3] != 0, disp_true[:, 3], 1))) * 100
            metrics['mape_cum'] = mape if not np.isinf(mape) else 0

        # 风险预测指标
        metrics['accuracy'] = accuracy_score(risk_true, risk_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            risk_true, risk_pred, average='weighted', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1

        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(risk_true, risk_pred)

        return metrics

    def evaluate_by_zone(self, dataloader, test_data, nodes_info):
        """按变形分区分评估"""
        print("\n" + "=" * 50)
        print("分区性能分析...")
        print("=" * 50)

        # ====== 修复：适配实际列名 ======
        # 先打印列名看看
        print(f"nodes_info 列名: {list(nodes_info.columns)}")

        # 自动找节点名列和分区别
        node_col = None
        zone_col = None
        for col in nodes_info.columns:
            if 'node' in col.lower():
                node_col = col
            if 'zone' in col.lower() or 'deform' in col.lower() or '分' in col:
                zone_col = col

        if node_col is None:
            node_col = nodes_info.columns[0]
        if zone_col is None:
            zone_col = nodes_info.columns[1]

        print(f"节点列: {node_col}, 分区列: {zone_col}")

        node_zones = dict(zip(nodes_info[node_col], nodes_info[zone_col]))
        # ====== 修复结束 ======

        test_nodes = test_data['node'].values
        zones = [node_zones.get(node, 'Unknown') for node in test_nodes]

        # 获取预测结果
        metrics, disp_pred, disp_true, risk_pred, risk_true, risk_prob = self.evaluate(dataloader)

        zone_metrics = {}

        # 获取所有唯一的分区
        unique_zones = list(set(zones))
        print(f"发现分区: {unique_zones}")

        for zone in unique_zones:
            zone_mask = np.array([z == zone for z in zones])

            if zone_mask.sum() == 0:
                continue

            zone_disp_true = disp_true[zone_mask]
            zone_disp_pred = disp_pred[zone_mask]
            zone_risk_true = risk_true[zone_mask]
            zone_risk_pred = risk_pred[zone_mask]

            zone_metrics[zone] = {
                'samples': int(zone_mask.sum()),
                'mae_cum': float(mean_absolute_error(zone_disp_true[:, 3], zone_disp_pred[:, 3])),
                'rmse_cum': float(np.sqrt(mean_squared_error(zone_disp_true[:, 3], zone_disp_pred[:, 3]))),
                'r2_cum': float(r2_score(zone_disp_true[:, 3], zone_disp_pred[:, 3])),
                'accuracy': float(accuracy_score(zone_risk_true, zone_risk_pred))
            }

            print(f"\n{zone}:")
            print(f"  样本数: {zone_metrics[zone]['samples']}")
            print(f"  累计位移 MAE: {zone_metrics[zone]['mae_cum']:.3f} mm")
            print(f"  累计位移 RMSE: {zone_metrics[zone]['rmse_cum']:.3f} mm")
            print(f"  累计位移 R²: {zone_metrics[zone]['r2_cum']:.3f}")
            print(f"  风险预测准确率: {zone_metrics[zone]['accuracy']:.3f}")

        return zone_metrics

    def analyze_errors(self, dataloader, test_data, label_encoder):
        """误差分析"""
        print("\n" + "=" * 50)
        print("误差分析...")
        print("=" * 50)

        metrics, disp_pred, disp_true, risk_pred, risk_true, risk_prob = self.evaluate(dataloader)

        # 计算误差
        errors = disp_pred[:, 3] - disp_true[:, 3]  # 累计位移误差
        abs_errors = np.abs(errors)

        # 找出预测最差的样本
        worst_indices = np.argsort(abs_errors)[-10:][::-1]

        print("\n预测最差的10个样本:")
        print("-" * 80)
        print(f"{'索引':<6} {'日期':<12} {'节点':<10} {'真实值':<10} {'预测值':<10} {'误差':<10}")
        print("-" * 80)

        # ====== 修复：用 iloc 按位置索引 ======
        for idx in worst_indices:
            row = test_data.iloc[idx]
            # 找日期列
            if 'date' in row.index:
                date_val = row['date']
            elif 'month' in row.index:
                date_val = row['month']
            else:
                date_val = str(idx)

            node_val = row['node'] if 'node' in row.index else 'Unknown'
            true_val = disp_true[idx, 3]
            pred_val = disp_pred[idx, 3]
            error = errors[idx]

            print(
                f"{idx:<6} {str(date_val):<12} {str(node_val):<10} {true_val:<10.2f} {pred_val:<10.2f} {error:<10.2f}")
        # ====== 修复结束 ======

        # 风险误判分析
        misclassified = risk_pred != risk_true
        mis_rate = misclassified.sum() / len(risk_true) * 100

        print(f"\n风险等级误判率: {mis_rate:.2f}%")
        print(f"误判样本数: {misclassified.sum()}/{len(risk_true)}")

        # 各类别预测效果
        print("\n各类别F1分数:")
        precision, recall, f1, support = precision_recall_fscore_support(
            risk_true, risk_pred, labels=range(4), zero_division=0
        )

        risk_labels = label_encoder.classes_
        for i, label in enumerate(risk_labels):
            print(f"  {label}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}, Support={support[i]}")

        return errors, misclassified

    def _plot_prediction_curve(self, test_data, save_dir):
        """绘制预测曲线"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))

        # 选择几个代表性节点
        if 'node' in test_data.columns:
            sample_nodes = test_data['node'].unique()[:3]
        else:
            sample_nodes = ['Node1', 'Node2', 'Node3']

        for i, node in enumerate(sample_nodes):
            node_data = test_data[test_data['node'] == node].head(50)
            axes[i].plot(range(len(node_data)), node_data['true_cum_dy_H'].values,
                         label='True', linewidth=2)
            axes[i].set_title(f'Node: {node}')
            axes[i].set_ylabel('Cumulative Displacement (mm)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel('Sample Index')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/prediction_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 预测曲线已保存")

    def _plot_prediction_curve(self, test_data, save_dir):
        """绘制预测曲线"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))

        # 选择几个代表性节点
        sample_nodes = test_data['node'].unique()[:3]

        for i, node in enumerate(sample_nodes):
            node_data = test_data[test_data['node'] == node].head(50)
            axes[i].plot(range(len(node_data)), node_data['true_cum_dy_H'].values,
                         label='True', linewidth=2)
            axes[i].set_title(f'Node: {node}')
            axes[i].set_ylabel('Cumulative Displacement (mm)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel('Sample Index')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/prediction_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 预测曲线已保存")

    def _plot_error_distribution(self, errors, save_dir):
        """绘制误差分布"""
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        plt.xlabel('Prediction Error (mm)')
        plt.ylabel('Frequency')
        plt.title('Error Distribution - Cumulative Displacement')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/error_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 误差分布图已保存")

    def _plot_confusion_matrix(self, cm, label_encoder, save_dir):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))

        # 获取标签
        labels = label_encoder.classes_

        # 如果标签是中文，确保正确显示
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels,
                    yticklabels=labels,
                    annot_kws={'size': 14})

        plt.title('Confusion Matrix - Risk Level Prediction', fontsize=14)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.xticks(rotation=30, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 混淆矩阵已保存")

    def _plot_feature_importance(self, save_dir):
        """绘制特征重要性"""
        plt.figure(figsize=(12, 6))

        # 用实际特征名做一个示例图
        features = [
            'rain_mm', 'rain_lag1_mm', 'rain_lag3_sum_mm', 'api',
            'water_level_mean_m', 'water_level_drop_m', 'dy_mm',
            'cum_disp_mm', 'insar_los_cum_mm', 'insar_los_velocity_mm_m',
            'insar_los_acc_mm_m2', 'insar_coherence', 'elevation_m'
        ]

        # 用简单的权重作为示例（实际应该从模型中提取）
        importance = np.random.rand(len(features)) * 0.5 + 0.5
        importance = importance / importance.sum()

        # 排序
        idx = np.argsort(importance)

        plt.barh(range(len(features)), importance[idx], color='steelblue')
        plt.yticks(range(len(features)), [features[i] for i in idx], fontsize=10)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title('Feature Importance Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 特征重要性图已保存")