# lab4/utils.py  – v4  (fixed 8-class label parsing, safe confusion-matrix)
import os, re
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score, roc_curve, auc,
                             ConfusionMatrixDisplay)
from sklearn.preprocessing import label_binarize

# -----------------------------------------------------------
# 设备选择
# -----------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# -----------------------------------------------------------
# 逐行数字解析（不依赖 pandas）
# -----------------------------------------------------------
def _read_single_csv(path: str, min_numeric_per_row: int = 4) -> np.ndarray:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip().replace("\t", ",")
            if not line:
                continue

            nums = []
            for tok in line.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    nums.append(float(tok))
                except ValueError:
                    continue

            if len(nums) >= min_numeric_per_row:
                rows.append(nums)

    if not rows:
        raise ValueError(f"[ERROR] {path} 内未找到任何数字行！")

    max_cols = max(len(r) for r in rows)
    data = np.zeros((len(rows), max_cols), dtype=np.float32)
    for i, r in enumerate(rows):
        data[i, :len(r)] = r

    # 若首列严格递增（帧号 / 时间戳）则删除
    if data.shape[1] > 1 and np.all(np.diff(data[:, 0]) > 0):
        data = data[:, 1:]

    return data

# -----------------------------------------------------------
# 固定 8 类动作标签解析：action001 → 0, …, action008 → 7
# -----------------------------------------------------------
_action_pat = re.compile(r"action(\d{3})", re.I)

def _extract_action_id(fname: str) -> int:
    m = _action_pat.search(fname)
    if not m:
        return -1
    val = int(m.group(1))
    if 1 <= val <= 8:
        return val - 1
    return -1    # 非 1-8 都舍弃

# -----------------------------------------------------------
# 数据集
# -----------------------------------------------------------
class SkeletonDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths):
        super().__init__()
        arrays, labels = [], []

        for p in file_paths:
            lab = _extract_action_id(os.path.basename(p))
            if lab == -1:
                continue  # 舍弃非法文件
            arrays.append(_read_single_csv(p))
            labels.append(lab)

        if not arrays:
            raise RuntimeError("[ERROR] 数据集中没有合法的 8 类样本！")

        self.max_len  = max(a.shape[0] for a in arrays)
        self.max_dim  = max(a.shape[1] for a in arrays)
        self.samples, self.labels = [], []

        for a, l in zip(arrays, labels):
            # 帧长 padding
            if a.shape[0] < self.max_len:
                pad = np.zeros((self.max_len - a.shape[0], a.shape[1]), np.float32)
                a = np.vstack([a, pad])
            # 特征维 padding
            if a.shape[1] < self.max_dim:
                pad = np.zeros((self.max_len, self.max_dim - a.shape[1]), np.float32)
                a = np.hstack([a, pad])

            self.samples.append(torch.tensor(a, dtype=torch.float32))
            self.labels.append(torch.tensor(l, dtype=torch.long))

        self.feature_dim = self.max_dim

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

# -----------------------------------------------------------
# 评价指标
# -----------------------------------------------------------
def evaluate_metrics(y_true, y_pred):
    acc = (y_true == y_pred).mean()
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, prec, rec, f1

# -----------------------------------------------------------
# 可视化
# -----------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    present = sorted(set(y_true) | set(y_pred))  # 真正出现的类别
    cm = confusion_matrix(y_true, y_pred, labels=present)

    disp_labels = [class_names[i] for i in present]
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=disp_labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path); plt.close(fig)
    else:
        plt.show()

def plot_roc(y_true, y_score, class_names, save_path=None):
    n_cls = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_cls)))

    fpr = {}; tpr = {}; roc_auc = {}
    for i in range(n_cls):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(5,5))
    for i, clr in enumerate(["C"+str(i) for i in range(n_cls)]):
        plt.plot(fpr[i], tpr[i], color=clr,
                 label=f"{class_names[i]} (AUC={roc_auc[i]:.2f})")
    plt.plot([0,1], [0,1], "k--", lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
    plt.legend(); plt.tight_layout()
    if save_path:
        plt.savefig(save_path); plt.close()
    else:
        plt.show()
