#!/usr/bin/env python
# coding: utf-8
"""
Iris — Missing-Value Strategies & Model Comparison  ✅ 完整版
----------------------------------------------------------------------
• 数据源      : ~/Desktop/AILab/lab2/iris/iris.data  (或 bezdekIris.data)
• 缺失值策略  : drop / mean / median / knn
• 模型        : LogisticRegression / DecisionTreeClassifier / SVC / KNN
• 评估        : accuracy, weighted-F1, confusion-matrix  (全部可视化)
• 可视化      : 英文柱状图 + 每个模型-策略的混淆矩阵热力图
• 输出目录    : ./figures  （脚本自动创建，图片自动命名保存）
"""

from __future__ import annotations

import pathlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ————————————— 全局超参 ————————————— #
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
MISSING_RATIO: float = 0.1  # >0 可手动注入缺失，验证策略差异
DATA_DIR = pathlib.Path("~/Desktop/AILab/lab2/iris").expanduser()
DATA_FILE = DATA_DIR / "iris.data"  # 如需用 bezdekIris.data，可改此行
FIG_DIR = pathlib.Path("./figures")

# ————————————— 工具函数 ————————————— #
def load_iris_df_from_csv(path: pathlib.Path) -> pd.DataFrame:
    """读取本地 CSV 并生成数值标签"""
    col_names = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target_name",
    ]
    df = pd.read_csv(path, header=None, names=col_names).dropna(how="all")
    classes = sorted(df["target_name"].unique())
    mapping = {cls: idx for idx, cls in enumerate(classes)}
    df["target"] = df["target_name"].map(mapping)
    return df


def inject_random_missing(df: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """随机掩盖一定比例特征值（演示用）"""
    if ratio <= 0.0:
        return df
    rng = np.random.default_rng(RANDOM_STATE)
    mask = rng.random(df.shape) < ratio
    feature_cols = df.columns[:4]
    df_miss = df.copy()
    df_miss.loc[:, feature_cols] = df_miss.loc[:, feature_cols].mask(mask[:, :4])
    return df_miss


def apply_missing_strategy(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """根据策略处理缺失值并返回新数据集"""
    if strategy == "drop":
        tr_mask = ~np.isnan(X_train).any(axis=1)
        te_mask = ~np.isnan(X_test).any(axis=1)
        return X_train[tr_mask], X_test[te_mask], y_train[tr_mask], y_test[te_mask]

    if strategy == "mean":
        imp = SimpleImputer(strategy="mean")
    elif strategy == "median":
        imp = SimpleImputer(strategy="median")
    elif strategy == "knn":
        imp = KNNImputer(n_neighbors=5)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return (
        imp.fit_transform(X_train),
        imp.transform(X_test),
        y_train,
        y_test,
    )


def ensure_fig_dir() -> None:
    """保证图片输出目录存在"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)


# ————————————— 训练 + 评估 ————————————— #
def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    strategy: str,
    class_labels: list[str],
) -> list[dict]:
    """训练四个模型并保存混淆矩阵热力图"""
    models: dict[str, object] = {
        "LogReg": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    results: list[dict] = []
    for name, clf in models.items():
        print(f"[INFO] Fitting {name} …", end=" ", flush=True)
        clf.fit(X_train, y_train)
        print("done.")

        preds = clf.predict(X_test)
        print(f"\nClassification report for {name} ({strategy}):\n"
              f"{classification_report(y_test, preds, digits=4)}")

        # --- 保存混淆矩阵 ---
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        fig, ax = plt.subplots(figsize=(4, 4))
        disp.plot(
            cmap="Blues",
            ax=ax,
            colorbar=False,
            values_format="d",
        )
        ax.set_title(f"Confusion Matrix — {name} ({strategy})")
        fig.tight_layout()
        save_path = FIG_DIR / f"confmat_{strategy}_{name}.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

        results.append(
            {
                "model": name,
                "strategy": strategy,
                "accuracy": accuracy_score(y_test, preds),
                "f1": f1_score(y_test, preds, average="weighted"),
            }
        )
    return results


# ————————————— 可视化 ————————————— #
def plot_bar(df: pd.DataFrame, metric: str) -> None:
    """绘制并保存 Accuracy / F1 柱状比较图"""
    pivot = df.pivot(index="model", columns="strategy", values=metric)
    fig, ax = plt.subplots(figsize=(6, 4))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(f"{metric.upper()} Comparison (Missing-Value Strategies)")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0, 1)
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.legend(title="Strategy")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{metric}_comparison.png", dpi=300)
    plt.close(fig)


# ————————————— 主流程 ————————————— #
def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    ensure_fig_dir()

    # 1. 载入本地 CSV
    df_raw = load_iris_df_from_csv(DATA_FILE)
    df = inject_random_missing(df_raw, ratio=MISSING_RATIO)

    # 2. train / test split
    X = df.iloc[:, :4].to_numpy()
    y = df["target"].to_numpy()
    class_labels = df["target_name"].unique().tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 3. 遍历缺失值策略
    strategies = ["drop", "mean", "median", "knn"]
    all_records: list[dict] = []

    for strat in strategies:
        print(f"\n================ Strategy = {strat.upper()} ================")
        Xt, Xv, yt, yv = apply_missing_strategy(
            X_train,
            X_test,
            y_train,
            y_test,
            strategy=strat,
        )
        if Xv.size == 0:
            print(f"[WARN] strategy '{strat}': empty test set after dropping — skipped.")
            continue

        scaler = StandardScaler()
        Xt_s, Xv_s = scaler.fit_transform(Xt), scaler.transform(Xv)

        records = evaluate_models(
            Xt_s,
            yt,
            Xv_s,
            yv,
            strategy=strat,
            class_labels=class_labels,
        )
        all_records.extend(records)

    # 4. 汇总结果表
    result_df = (
        pd.DataFrame(all_records)
        .sort_values(["strategy", "model"])
        .reset_index(drop=True)
    )
    print("\n===== Consolidated Metrics =====")
    print(result_df.to_string(index=False))

    # 5. 绘制并保存柱状图
    for metric in ["accuracy", "f1"]:
        plot_bar(result_df, metric)

    print(f"\n✅ All figures saved to: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
