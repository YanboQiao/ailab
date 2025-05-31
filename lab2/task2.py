#!/usr/bin/env python
# coding: utf-8
"""
Mall Customers — end-to-end clustering workflow
------------------------------------------------------------------
• Load + explore + clean + encode + scale
• Dimensional reduction: PCA (2D) & t-SNE (2D)
• Clustering: KMeans / Agglomerative / DBSCAN
• Internal metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin
• 可视化 (英文) : 7 figures  自动保存至 ./figures
"""

from __future__ import annotations

import pathlib
import warnings
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import (AgglomerativeClustering, DBSCAN,
                             KMeans)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (calinski_harabasz_score,
                             davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------- config ---------------------------- #
RANDOM_STATE: int = 42
DATA_FILE = pathlib.Path("Mall_Customers.csv")
GITHUB_RAW = (
    "https://raw.githubusercontent.com/Karansingh1221/"
    "Mall_Customer_dataset/main/Mall_Customers.csv"
)
FIG_DIR = pathlib.Path("./figures2")

# ---------------------------- helpers --------------------------- #
def download_data(url: str, dst: pathlib.Path) -> None:
    print(f"[INFO] downloading dataset from\n{url}")
    dst.write_bytes(pathlib.request.urlopen(url).read())  # noqa: S310


def load_data(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        from urllib.request import urlopen

        print("[INFO] dataset not found, downloading ...")
        path.write_bytes(urlopen(GITHUB_RAW).read())  # noqa: S310
    df = pd.read_csv(path)
    df.columns = [
        "CustomerID",
        "Gender",
        "Age",
        "AnnualIncome",
        "SpendingScore",
    ]
    return df


def basic_eda(df: pd.DataFrame) -> None:
    print("\n=== Data overview ===")
    print(df.head(), "\n")
    print(df.info())
    print("\nMissing values per column:\n", df.isnull().sum())


def treat_outliers(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Winsorise extreme outliers beyond 3×IQR."""
    for col in cols:
        q1, q3 = np.percentile(df[col], [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        df[col] = np.clip(df[col], lower, upper)
    return df


def encode_and_scale(df: pd.DataFrame) -> Tuple[np.ndarray, list[str]]:
    """Label-encode ‘Gender’ & scale numeric features."""
    num_cols = ["Age", "AnnualIncome", "SpendingScore"]
    cat_cols = ["Gender"]

    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    feature_cols = num_cols + cat_cols
    X = df[feature_cols].to_numpy(dtype=float)
    return X, feature_cols


def reduce_dimensions(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    tsne = TSNE(
        n_components=2,
        random_state=RANDOM_STATE,
        perplexity=30,
        n_iter=1000,
        learning_rate="auto",
    )
    X_tsne = tsne.fit_transform(X)
    return X_pca, X_tsne


def ensure_fig_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------- visual utilities ------------------ #
def elbow_and_silhouette(X: np.ndarray) -> int:
    """绘制肘部法则 & 轮廓系数曲线，返回最佳 k（基于 silhouette 最大）"""
    inertias, sil_scores = [], []
    k_range = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE).fit(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, km.labels_))

    # Elbow
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(k_range), inertias, marker="o")
    ax.set_title("Elbow Method (KMeans)")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "elbow.png", dpi=300)
    plt.close(fig)

    # Silhouette
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(k_range), sil_scores, marker="o")
    ax.set_title("Silhouette Scores vs. k")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette score")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "silhouette_curve.png", dpi=300)
    plt.close(fig)

    best_k = k_range[int(np.argmax(sil_scores))]
    print(f"[INFO] Best k by silhouette = {best_k}")
    return best_k


def plot_feature_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col in zip(axes, ["Age", "AnnualIncome", "SpendingScore"]):
        sns.histplot(df[col], bins=15, ax=ax)
        ax.set_title(f"{col} distribution")
    fig.suptitle("Feature Distributions (after scaling)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_hist.png", dpi=300)
    plt.close(fig)


def plot_scatter_income_vs_score(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["AnnualIncome"], df["SpendingScore"], edgecolor="k")
    ax.set_title("Annual Income vs. Spending Score")
    ax.set_xlabel("Annual Income (std)")
    ax.set_ylabel("Spending Score (std)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "income_vs_score.png", dpi=300)
    plt.close(fig)


def plot_2d_clusters(
    X_2d: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray | None,
    algo_name: str,
    reducer_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        cmap="tab10",
        s=40,
        edgecolor="k",
    )
    if centroids is not None:
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="X",
            s=200,
            linewidths=2,
            edgecolors="k",
        )
    ax.set_title(f"{reducer_name} (2D) — clusters by {algo_name}")
    ax.set_xlabel(f"{reducer_name}-1")
    ax.set_ylabel(f"{reducer_name}-2")
    fig.tight_layout()
    fname = f"{reducer_name.lower()}_{algo_name.lower().split('(')[0]}.png"
    fig.savefig(FIG_DIR / fname, dpi=300)
    plt.close(fig)


def plot_dendrogram(X: np.ndarray) -> None:
    Z = linkage(X, method="ward")
    fig, ax = plt.subplots(figsize=(8, 4))
    dendrogram(Z, truncate_mode="level", p=5, ax=ax, color_threshold=0)
    ax.set_title("Hierarchical Clustering Dendrogram (truncated)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dendrogram.png", dpi=300)
    plt.close(fig)


# ---------------------------- clustering workflow --------------- #
def cluster_and_evaluate(
    X: np.ndarray,
    X_pca: np.ndarray,
    X_tsne: np.ndarray,
    k_opt: int,
) -> None:
    print(f"\n=== Clustering (k = {k_opt}) ===")
    models = {
        f"KMeans(k={k_opt})": KMeans(n_clusters=k_opt, random_state=RANDOM_STATE),
        f"Agglomerative(k={k_opt})": AgglomerativeClustering(n_clusters=k_opt),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    }

    for name, model in models.items():
        pred = model.fit_predict(X)
        # DBSCAN: silhouette 需 ≥2 个簇且不能全为 -1
        valid = len(set(pred)) > 2 or (len(set(pred)) == 2 and -1 not in pred)
        sil = silhouette_score(X, pred) if valid else np.nan
        ch = calinski_harabasz_score(X, pred)
        db = davies_bouldin_score(X, pred)
        print(f"{name:20s} | silhouette={sil:.3f} | CH={ch:.1f} | DB={db:.3f}")

        # 可视化 (PCA & t-SNE) — 只为 KMeans 保存质心
        cent = model.cluster_centers_ if hasattr(model, "cluster_centers_") else None
        plot_2d_clusters(X_pca, pred, cent, name, "PCA")
        plot_2d_clusters(X_tsne, pred, None, name, "tSNE")

    print("\n(↑ silhouette / CH 越大越好；DB 越小越好)")


# ---------------------------- main ------------------------------ #
def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    ensure_fig_dir()

    # 1. 载入与 EDA
    df = load_data(DATA_FILE)
    basic_eda(df)

    # 2. Outlier 处理
    df = treat_outliers(df, ["Age", "AnnualIncome", "SpendingScore"])

    # 3. Encode & scale
    X, _ = encode_and_scale(df)

    # 4. 可视化 (探索性)
    plot_scatter_income_vs_score(df)
    plot_feature_distributions(df)

    # 5. 维度缩减
    X_pca, X_tsne = reduce_dimensions(X)

    # 6. Elbow & Silhouette → 最优 k
    best_k = elbow_and_silhouette(X)

    # 7. 层次聚类树状图
    plot_dendrogram(X)

    # 8. 聚类 + 评估 + 结果可视化
    cluster_and_evaluate(X, X_pca, X_tsne, k_opt=best_k)

    print(f"\n✅ All figures saved to: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()