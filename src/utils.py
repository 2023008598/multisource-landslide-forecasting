import numpy as np
import pandas as pd
import torch
import random
import json
from config import Config


def set_seed(seed=None):
    """设置随机种子确保可复现性"""
    if seed is None:
        seed = Config().SEED

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"✓ 随机种子已设置: {seed}")


def save_metrics(metrics_dict, filepath):
    """保存评估指标到JSON文件"""

    # 转换numpy数组为列表
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    metrics_dict = convert(metrics_dict)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    print(f"✓ 评估指标已保存到 {filepath}")


def save_predictions(test_data, disp_pred, disp_true, risk_pred, risk_true,
                     risk_prob, label_encoder, filepath):
    """保存预测结果到CSV"""

    # ====== 修复：兼容不同列名 ======
    date_col = 'date' if 'date' in test_data.columns else 'month'
    node_col = 'node' if 'node' in test_data.columns else test_data.columns[0]
    # ====== 修复结束 ======

    results = pd.DataFrame({
        'date': test_data[date_col].values,
        'node': test_data[node_col].values,
        'horizon': 3,
        'true_dy_h1': disp_true[:, 0],
        'pred_dy_h1': disp_pred[:, 0],
        'true_dy_h2': disp_true[:, 1],
        'pred_dy_h2': disp_pred[:, 1],
        'true_dy_h3': disp_true[:, 2],
        'pred_dy_h3': disp_pred[:, 2],
        'true_cum_dy_H': disp_true[:, 3],
        'pred_cum_dy_H': disp_pred[:, 3],
        'true_max_dy_H': disp_true[:, 4],
        'pred_max_dy_H': disp_pred[:, 4],
        'true_avg_dy_H': disp_true[:, 5],
        'pred_avg_dy_H': disp_pred[:, 5],
        'true_label_future': label_encoder.inverse_transform(risk_true),
        'pred_label_future': label_encoder.inverse_transform(risk_pred),
        'confidence': np.max(risk_prob, axis=1)
    })

    results['error_cum'] = results['pred_cum_dy_H'] - results['true_cum_dy_H']
    results['abs_error_cum'] = np.abs(results['error_cum'])

    results.to_csv(filepath, index=False)
    print(f"✓ 预测结果已保存到 {filepath}")

    return results


def print_model_summary(model):
    """打印模型结构摘要"""
    print("\n" + "=" * 50)
    print("模型结构摘要")
    print("=" * 50)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"\n模型结构:")
    print(model)

    # 打印各层参数量
    print("\n各模块参数量:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params:,}")


def check_gpu_availability():
    """检查GPU可用性"""
    print("\n" + "=" * 50)
    print("硬件环境检查")
    print("=" * 50)

    if torch.cuda.is_available():
        print(f"✓ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        print(f"  当前GPU内存: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
    else:
        print("⚠ GPU不可用，将使用CPU训练")
        print("  训练时间可能较长")

    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_output_dirs():
    """创建输出目录"""
    import os

    dirs = [
        'outputs/models',
        'outputs/results',
        'outputs/figures'
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)

    print("✓ 输出目录已创建")