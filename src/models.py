import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class MultiTaskGRU(nn.Module):
    """基线模型：多任务GRU"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(MultiTaskGRU, self).__init__()

        config = Config()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 位移回归头
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, config.HORIZON + 3)
        )

        # 风险分类头
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        # 维度修复：确保输入是3维 (batch, seq, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, features) -> (batch, 1, features)

        gru_out, _ = self.gru(x)  # (batch, 1, hidden*2)
        last_hidden = gru_out[:, -1, :]  # (batch, hidden*2)

        disp_pred = self.regression_head(last_hidden)
        risk_pred = self.classification_head(last_hidden)

        return disp_pred, risk_pred


class AttentionGRU(nn.Module):
    """改进模型：带注意力机制的GRU"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(AttentionGRU, self).__init__()

        config = Config()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, config.HORIZON + 3)
        )

        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        # 维度修复：确保输入是3维
        if x.dim() == 2:
            x = x.unsqueeze(1)

        gru_out, _ = self.gru(x)  # (batch, 1, hidden*2)

        # 注意力权重
        attn_weights = self.attention(gru_out)  # (batch, 1, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        # 加权求和
        context = torch.sum(attn_weights * gru_out, dim=1)  # (batch, hidden*2)

        disp_pred = self.regression_head(context)
        risk_pred = self.classification_head(context)

        return disp_pred, risk_pred


class LSTMModel(nn.Module):
    """LSTM模型（可选）"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()

        config = Config()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, config.HORIZON + 3)
        )

        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]

        disp_pred = self.regression_head(last_hidden)
        risk_pred = self.classification_head(last_hidden)

        return disp_pred, risk_pred


class TCNModel(nn.Module):
    """简单的TCN模型（可选）"""

    def __init__(self, input_size, hidden_size=64, dropout=0.2):
        super(TCNModel, self).__init__()

        config = Config()

        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, config.HORIZON + 3)
        )

        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        # TCN需要 (batch, features, seq)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, features) -> (batch, 1, features)

        x = x.transpose(1, 2)  # (batch, features, 1)

        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))

        x = self.global_pool(x).squeeze(-1)

        disp_pred = self.regression_head(x)
        risk_pred = self.classification_head(x)

        return disp_pred, risk_pred