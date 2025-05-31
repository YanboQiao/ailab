# lab4/main.py  -----------------------------------------------------------
#  Skeleton CSV  ➜  Deep Bi-LSTM + Self-Attention 动作识别
#  * 随机划分 9:1
#  * 自动保存 best / latest 模型
# ------------------------------------------------------------------------

import os, random, argparse, re, pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model  import ActionClassifier
from utils  import (get_device, SkeletonDataset, evaluate_metrics,
                    plot_confusion_matrix, plot_roc)

# ----------------------------------------------------------------------
def random_split(csv_list, train_ratio=0.9, seed=None):
    """彻底随机打乱后切分"""
    if seed is not None:
        random.seed(seed)
    files = csv_list.copy()
    random.shuffle(files)
    cut = int(train_ratio * len(files))
    return files[:cut], files[cut:]

def pad_feature_dim(ds, target):
    if ds.feature_dim == target:
        return
    for i in range(len(ds.samples)):
        diff = target - ds.samples[i].shape[1]
        if diff:
            pad = torch.zeros(ds.samples[i].shape[0], diff)
            ds.samples[i] = torch.cat([ds.samples[i], pad], dim=1)
    ds.feature_dim = target

# ----------------------------------------------------------------------
def main():
    # ---------------- CLI ----------------
    ap = argparse.ArgumentParser("Skeleton Action Recognition (random split)")
    ap.add_argument("--data-dir",  type=str, default="skeleton")
    ap.add_argument("--epochs",    type=int, default=32)
    ap.add_argument("--batch-size",type=int, default=64)
    ap.add_argument("--lr",        type=float, default=4e-3)
    ap.add_argument("--seed",      type=int, default=None,
                    help="随机种子 (设定可复现)")
    ap.add_argument("--ckpt-dir",  type=str, default="checkpoints",
                    help="模型保存文件夹")
    args = ap.parse_args()

    pathlib.Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    # ---------------- 1. 收集 CSV ----------------
    csv_files = [os.path.join(args.data_dir, f)
                 for f in os.listdir(args.data_dir)
                 if f.lower().endswith(".csv")]
    if not csv_files:
        print("[ERROR] 未找到任何 CSV 文件"); return

    train_files, test_files = random_split(csv_files, 0.9, args.seed)
    print(f"[INFO] total {len(csv_files)} – "
          f"{len(train_files)} train / {len(test_files)} test")

    # ---------------- 2. 数据集 ----------------
    train_ds = SkeletonDataset(train_files)
    test_ds  = SkeletonDataset(test_files)

    feat_dim = max(train_ds.feature_dim, test_ds.feature_dim)
    pad_feature_dim(train_ds, feat_dim)
    pad_feature_dim(test_ds,  feat_dim)

    train_ld = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True)
    test_ld  = torch.utils.data.DataLoader(
        test_ds,  batch_size=args.batch_size, shuffle=False)

    # ---------------- 3. 标签压缩 ----------------
    org_labels = [int(l) for l in train_ds.labels + test_ds.labels]
    uniq       = sorted(set(org_labels))
    old2new    = {o:i for i, o in enumerate(uniq)}
    for ds in (train_ds, test_ds):
        ds.labels = [torch.tensor(old2new[int(l)]) for l in ds.labels]

    num_cls   = len(uniq)
    cls_names = [str(o) for o in uniq]

    # ---------------- 4. 模型 ----------------
    device = get_device()
    model = ActionClassifier(
        input_dim   = feat_dim,
        hidden_dim  = 256,
        num_layers  = 2,
        num_classes = num_cls).to(device)

    loss_fn  = nn.CrossEntropyLoss()
    optimzer = optim.AdamW(model.parameters(), lr=args.lr)

    # ---------------- 5. 训练 & 自动保存 ----------------
    best_acc = 0.0
    best_path   = os.path.join(args.ckpt_dir, "best.pt")
    latest_path = os.path.join(args.ckpt_dir, "latest.pt")

    for ep in range(1, args.epochs + 1):
        model.train(); run_loss = 0.0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)

            optimzer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimzer.step()
            run_loss += loss.item() * x.size(0)

        # ------ 保存最新权重 ------
        torch.save(model.state_dict(), latest_path)

        # ------ epoch 评估 ------
        model.eval(); correct = tot = 0
        with torch.no_grad():
            for x, y in test_ld:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                tot     += y.size(0)
        acc = correct / tot

        flag = ""
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)
            flag = "  [BEST ↓ saved]"
        print(f"Epoch {ep:02d}/{args.epochs}  "
              f"Loss {run_loss/len(train_ds):.4f}  "
              f"TestAcc {acc:.4f}{flag}")

    print(f"[INFO] best model saved to {best_path}")
    print(f"[INFO] latest model saved to {latest_path}")

    # ---------------- 6. 最终评估 & 绘图 ----------------
    #   * 使用「最佳权重」重新加载再测一次（可选）
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval(); P, L, Prob = [], [], []
    with torch.no_grad():
        for x, y in test_ld:
            x, y = x.to(device), y.to(device)
            out = model(x)
            P.extend(out.argmax(1).cpu().numpy())
            L.extend(y.cpu().numpy())
            Prob.extend(torch.softmax(out,1).cpu().numpy())

    P, L, Prob = map(np.array, (P, L, Prob))
    mask = ~np.isnan(Prob).any(axis=1)
    if mask.sum() < len(Prob):
        print(f"[WARN] {len(Prob)-mask.sum()} NaN 样本已剔除")
    P, L, Prob = P[mask], L[mask], Prob[mask]

    acc, pre, rec, f1 = evaluate_metrics(L, P)
    print(f"\n[FINAL] Acc:{acc:.4f}  Prec:{pre:.4f}  "
          f"Recall:{rec:.4f}  F1:{f1:.4f}")

    plot_confusion_matrix(L, P, cls_names, "confusion_matrix.png")
    plot_roc(L, Prob, cls_names, "roc_curves.png")
    print("[INFO] confusion_matrix.png / roc_curves.png 已保存")


if __name__ == "__main__":
    main()
