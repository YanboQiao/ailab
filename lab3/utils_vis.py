"""
训练过程可视化：生成一张图展示多项指标
"""
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_history(
        history: Dict[str, List[float]],
        out_path: str | Path = "metrics.png",
        dpi: int = 150
) -> None:
    keys = list(history.keys())
    n_cols = 2
    n_rows = (len(keys) + 1) // n_cols
    plt.figure(figsize=(6 * n_cols, 3 * n_rows))

    for idx, k in enumerate(keys, 1):
        plt.subplot(n_rows, n_cols, idx)
        plt.plot(history[k], label=k, linewidth=1.2)
        plt.title(k)
        plt.xlabel("Epoch")
        plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
