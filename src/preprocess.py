import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import Config


class DataPreprocessor:
    """数据预处理器"""

    def __init__(self):
        self.config = Config()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_cols = None
        self.target_cols = None

    def split_by_time(self, samples):
        """按时间顺序划分数据集"""
        print("\n" + "=" * 50)
        print("按时间顺序划分数据集...")
        print("=" * 50)

        # date 列已在 data_loader 中创建好了
        samples = samples.sort_values('date')

        # 时间划分 (加-01变成完整日期)
        train_end = pd.to_datetime(self.config.TRAIN_END + '-01')
        val_start = pd.to_datetime(self.config.VAL_START + '-01')
        val_end = pd.to_datetime(self.config.VAL_END + '-01')
        test_start = pd.to_datetime(self.config.TEST_START + '-01')

        train_mask = samples['date'] <= train_end
        val_mask = (samples['date'] >= val_start) & (samples['date'] <= val_end)
        test_mask = samples['date'] >= test_start

        train_data = samples[train_mask].copy()
        val_data = samples[val_mask].copy()
        test_data = samples[test_mask].copy()

        print(f"训练集: {len(train_data)} 样本")
        if len(train_data) > 0:
            print(
                f"  时间范围: {train_data['date'].min().strftime('%Y-%m')} 到 {train_data['date'].max().strftime('%Y-%m')}")

        print(f"验证集: {len(val_data)} 样本")
        if len(val_data) > 0:
            print(
                f"  时间范围: {val_data['date'].min().strftime('%Y-%m')} 到 {val_data['date'].max().strftime('%Y-%m')}")

        print(f"测试集: {len(test_data)} 样本")
        if len(test_data) > 0:
            print(
                f"  时间范围: {test_data['date'].min().strftime('%Y-%m')} 到 {test_data['date'].max().strftime('%Y-%m')}")

        print("✓ 时间划分完成")

        return train_data, val_data, test_data

    def prepare_features_targets(self, data):
        """准备特征和目标变量"""
        exclude_cols = ['date', 'node', 'month', 'sample_id', 'window', 'horizon',
                        'zone', 'risk_rule',
                        'future_dy_h1', 'future_dy_h2', 'future_dy_h3',
                        'true_cum_dy_H', 'max_dy_future_H', 'avg_dy_future_H',
                        'true_label_future']

        self.feature_cols = [col for col in data.columns if col not in exclude_cols]

        self.target_cols = ['future_dy_h1', 'future_dy_h2', 'future_dy_h3',
                            'true_cum_dy_H', 'max_dy_future_H', 'avg_dy_future_H']

        print(f"特征数量: {len(self.feature_cols)}")
        print(f"特征列: {self.feature_cols}")
        print(f"目标列: {self.target_cols}")

        return self.feature_cols, self.target_cols

    def extract_features_targets(self, data):
        """提取特征和目标变量"""
        X = data[self.feature_cols].values
        y_disp = data[self.target_cols].values
        y_risk = self.label_encoder.fit_transform(data['true_label_future'])

        print(f"风险等级映射: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        return X, y_disp, y_risk

    def normalize_features(self, train_X, val_X=None, test_X=None):
        """标准化特征（仅在训练集上计算统计量）"""
        print("\n标准化特征（仅使用训练集统计量）...")

        self.scaler.fit(train_X)
        train_X_scaled = self.scaler.transform(train_X)

        val_X_scaled = None
        if val_X is not None and len(val_X) > 0:
            val_X_scaled = self.scaler.transform(val_X)
        else:
            val_X_scaled = np.array([])

        test_X_scaled = None
        if test_X is not None and len(test_X) > 0:
            test_X_scaled = self.scaler.transform(test_X)
        else:
            test_X_scaled = np.array([])

        print(f"✓ 训练集均值范围: [{self.scaler.mean_[:3].min():.2f}, {self.scaler.mean_[:3].max():.2f}]")

        return train_X_scaled, val_X_scaled, test_X_scaled

    def construct_lag_features(self, daily_data):
        """构造滞后特征（从日尺度数据）"""
        if daily_data is None:
            print("无日尺度数据，跳过滞后特征构造")
            return None

        print("\n构造滞后降雨和水位特征...")

        # 确保日期格式
        daily_data['date'] = pd.to_datetime(daily_data['date'])

        # 按月汇总
        daily_data['year_month'] = daily_data['date'].dt.to_period('M')

        monthly_summary = daily_data.groupby('year_month').agg({
            'rainfall': ['sum', 'mean', 'max'],
            'waterlevel': ['mean', 'min', 'max', 'std']
        }).reset_index()

        # 重命名列
        monthly_summary.columns = ['year_month', 'rain_sum', 'rain_mean', 'rain_max',
                                   'wl_mean', 'wl_min', 'wl_max', 'wl_std']

        # 构造滞后特征
        monthly_summary['rain_lag1'] = monthly_summary['rain_sum'].shift(1)
        monthly_summary['rain_lag2'] = monthly_summary['rain_sum'].shift(2)
        monthly_summary['rain_cum3'] = monthly_summary['rain_sum'].rolling(3).sum()

        # API指数（前期降雨指数，指数衰减）
        def calc_api(rain_series, k=0.85):
            api = np.zeros(len(rain_series))
            for i in range(1, len(rain_series)):
                api[i] = api[i - 1] * k + rain_series.iloc[i - 1]
            return api

        monthly_summary['api'] = calc_api(monthly_summary['rain_sum'])

        # 水位消落特征
        monthly_summary['wl_diff'] = monthly_summary['wl_max'] - monthly_summary['wl_min']
        monthly_summary['wl_drawdown'] = monthly_summary['wl_mean'].diff()

        print("✓ 滞后特征构造完成")
        return monthly_summary