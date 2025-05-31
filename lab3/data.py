"""
data.py
=======

提供 get_dataloaders()，完成：

1. 本地或在线下载 MNIST 数据集
2. 训练集常用数据增强（随机旋转 / 平移 / 轻微缩放）
3. 归一化 & 转张量
4. 返回训练 / 测试 DataLoader
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ------------------------------------------------------------
# 数据增强与 DataLoader
# ------------------------------------------------------------
def get_dataloaders(
    data_dir: str | Path = "./MNIST",
    batch_size: int = 128,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    参数
    ----
    data_dir   : MNIST 存放目录，若缺失会自动下载
    batch_size : 批大小
    num_workers: DataLoader 线程数（0 表示主进程加载）
    """
    data_dir = Path(data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    # ---- 训练集数据增强 ----
    train_tf = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=10,          # 随机旋转 ±10°
                translate=(0.1, 0.1),# 平移 ±10%
                scale=(0.9, 1.1),    # 缩放 0.9~1.1
            ),
            transforms.ToTensor(),                   # 0-1 归一化
            transforms.Normalize((0.1307,), (0.3081,))# 标准化
        ]
    )

    # ---- 测试集只做归一化 ----
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    )

    train_ds = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=train_tf,
    )
    test_ds = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=test_tf,
    )

    # Apple Silicon 上 pin_memory 不支持，故设 False
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_dl, test_dl