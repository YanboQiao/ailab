#!/usr/bin/env python3
"""可视化最佳模型在测试集上的预测结果（前 36 张）。

示例::
    python visualize.py --model-path runs/mnist_cnn/best_model.pt --data-dir ./MNIST
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CNNModel


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--n-samples", type=int, default=36)
    p.add_argument("--rows", type=int, default=6)
    return p.parse_args()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # transforms 与训练保持一致 (归一化)
    mean, std = 0.1307, 0.3081
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])
    test_ds = datasets.MNIST(str(args.data_dir), train=False, download=False, transform=test_tf)
    test_dl = DataLoader(test_ds, batch_size=args.n_samples, shuffle=False)

    model = CNNModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 取前 n-samples 张预测
    images, labels = next(iter(test_dl))
    images = images[: args.n_samples].to(device)
    labels = labels[: args.n_samples]
    with torch.no_grad():
        preds = model(images).argmax(dim=1).cpu()

    # 反归一化到 0‑1 便于显示
    images = images.cpu() * std + mean

    rows = args.rows
    cols = (args.n_samples + rows - 1) // rows

    plt.figure(figsize=(cols * 2, rows * 2))
    for idx in range(args.n_samples):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(images[idx][0], cmap="gray")
        plt.title(f"GT:{labels[idx]} / Pred:{preds[idx]}", fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()