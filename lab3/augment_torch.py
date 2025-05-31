# augment_torch.py
"""
PyTorch 数据增强 & DataLoader 构建
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

__all__ = ["get_dataloaders"]  # 显式导出

# ---------- 数据增强 ---------- #
def _get_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


# ---------- DataLoader ---------- #
def get_dataloaders(
    data_dir: str | Path = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    返回 train / val / test 三个 DataLoader
    """
    data_dir = Path(data_dir)
    train_full = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=_get_transforms(True)
    )

    val_size = int(len(train_full) * val_split)
    train_size = len(train_full) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        train_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    test_ds = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=_get_transforms(False)
    )

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_dl, val_dl, test_dl
