# model_torch.py
"""
PyTorch 版轻量级 CNN（MNIST）
"""

from __future__ import annotations
import torch
from torch import nn

__all__ = ["CNNPyTorch"]  # 显式导出，避免 import * 混乱


class CNNPyTorch(nn.Module):
    """
    2×Conv ➜ MaxPool ➜ 2×Conv ➜ MaxPool ➜ 2×FC
    参数量 ≈ 1.2 M，适合 MNIST
    """

    def __init__(
        self,
        channels: tuple[int, int] = (32, 64),
        fc_dim: int = 128,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        c1, c2 = channels

        self.features = nn.Sequential(
            # Block-1
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),               # 28×28 → 14×14
            nn.Dropout(0.25),

            # Block-2
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),               # 14×14 → 7×7
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fc_dim, num_classes),
        )

    # forward 允许 torchscript 检查；按约定返回 logits
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        return self.classifier(x)
