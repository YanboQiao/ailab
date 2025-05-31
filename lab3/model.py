"""
model.py
========
MNIST 专用轻量级卷积神经网络，参数 < 1.2 M。
两次池化后特征图大小固定为 7×7 → 与 fc1 维度严格匹配。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """Topology:
    conv1 (1→32, 3×3) → ReLU → pool1
    conv2 (32→64, 3×3) → ReLU → pool2
    dropout1 (0.25)
    flatten
    fc1 (64·7·7 → 128) → ReLU
    dropout2 (0.5)
    fc2 (128 → 10)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 28×28
        self.pool1 = nn.MaxPool2d(2, 2)                           # 14×14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14×14
        self.pool2 = nn.MaxPool2d(2, 2)                           # 7×7
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)          # (N, 64·7·7)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)

    # --------------------------------------------------------
    # 权重初始化
    # --------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)