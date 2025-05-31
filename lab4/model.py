# lab4/model.py
# -------------------------------------------------------------
# 深层双向 LSTM + Attention 适用于 Skeleton-CSV 动作识别
# -------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    对时间维 hidden states 做 soft attention：
        α_t = softmax( wᵀ h_t )
        context = Σ α_t * h_t
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h):                 # h: (B, T, H)
        α = self.score(h).squeeze(-1)     # (B, T)
        α = torch.softmax(α, dim=1)       # 注意力权重
        context = torch.sum(h * α.unsqueeze(-1), dim=1)  # (B, H)
        return context


class ActionClassifier(nn.Module):
    """
    input_dim  : 每帧特征维度
    hidden_dim : LSTM 隐状态（单向）维度
    num_layers : LSTM 堆叠层数
    num_classes: 动作类别数
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 num_classes: int = 8,
                 dropout: float = 0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0)

        self.attn_pool = AttentionPooling(hidden_dim * 2)   # 双向 → 2H

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self._init_weights()

    # ---------------------------------------------------------
    def _init_weights(self):
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.constant_(p, 0.0)

    # ---------------------------------------------------------
    def forward(self, x):                # x: (B, T, F)
        h, _ = self.lstm(x)              # (B, T, 2H)
        context = self.attn_pool(h)      # (B, 2H)
        logits = self.fc(context)        # (B, C)
        return logits
