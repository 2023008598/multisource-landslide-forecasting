import pandas as pd
import numpy as np
from config import Config


class DataLoader:
    """数据加载器"""

    def __init__(self):
        self.config = Config()
        self.samples = None
        self.features = None
        self.nodes = None
        self.daily_data = None

    def load_all_data(self):
        """加载所有数据文件"""
        print("=" * 50)
        print("加载数据文件...")
        print("=" * 50)

        self.samples = pd.read_csv(f"{self.config.DATA_DIR}/samples_H3_W12.csv")
        self.features = pd.read_csv(f"{self.config.DATA_DIR}/monthly_multisource_features.csv")
        self.nodes = pd.read_csv(f"{self.config.DATA_DIR}/node_info.csv")

        try:
            self.daily_data = pd.read_csv(f"{self.config.DATA_DIR}/rainfall_waterlevel_daily.csv")
            print("✓ 日尺度数据加载成功")
        except:
            print("⚠ 日尺度数据未找到，跳过")
            self.daily_data = None

        print(f"✓ 样本数据: {self.samples.shape}")
        print(f"✓ 月特征数据: {self.features.shape}")
        print(f"✓ 节点信息: {self.nodes.shape}")

        # ====== 修复 month 列格式 Dec-18 -> 2018-12 ======
        # 自定义解析函数
        def parse_month(m):
            month_map = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                         'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                         'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
            parts = str(m).split('-')
            if len(parts) == 2:
                mon_abbr = parts[0].strip()
                yr = parts[1].strip()
                if len(yr) == 2:
                    yr = '20' + yr
                mon_num = month_map.get(mon_abbr, '01')
                return pd.to_datetime(f"{yr}-{mon_num}-01")
            return pd.NaT

        self.samples['date'] = self.samples['month'].apply(parse_month)
        # ====== 修复结束 ======

        print(f"samples列名: {list(self.samples.columns)}")
        print(f"数据时间范围: {self.samples['date'].min()} 到 {self.samples['date'].max()}")
        print(f"监测节点数量: {self.samples['node'].nunique()}")
        print(f"节点列表: {list(self.samples['node'].unique())}")

        return self.samples, self.features, self.nodes

    def get_node_info(self):
        """获取节点信息"""
        if self.nodes is None:
            self.load_all_data()
        return self.nodes

    def get_feature_names(self):
        """获取特征名称列表"""
        if self.samples is None:
            self.load_all_data()

        # 排除非特征列
        exclude_cols = ['date', 'node', 'year', 'month',
                        'future_dy_h1', 'future_dy_h2', 'future_dy_h3',
                        'true_cum_dy_H', 'max_dy_future_H', 'avg_dy_future_H',
                        'true_label_future']

        feature_cols = [col for col in self.samples.columns if col not in exclude_cols]
        return feature_cols

    def get_target_names(self):
        """获取目标变量名称"""
        target_cols = ['future_dy_h1', 'future_dy_h2', 'future_dy_h3',
                       'true_cum_dy_H', 'max_dy_future_H', 'avg_dy_future_H']
        return target_cols

    def get_risk_labels(self):
        """获取风险标签"""
        if self.samples is None:
            self.load_all_data()
        return self.samples['true_label_future'].unique()