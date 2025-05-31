"""
统一训练入口
用法示例：
    python train.py --framework torch --epochs 20
    python train.py --framework tf    --epochs 20
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List

# ---------- 公共函数 -------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MNIST CNN Training")
    p.add_argument("--framework", choices=["torch", "tf"], default="torch")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out-dir", type=str, default="runs")
    # 让 Windows 用户可以在命令行覆盖 num_workers
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) / args.framework
    out_dir.mkdir(parents=True, exist_ok=True)

    # ========== PyTorch ==========
    if args.framework == "torch":
        import torch
        from torch import nn, optim
        from sklearn.metrics import precision_score, recall_score, f1_score

        from augment_torch import get_dataloaders
        from model_torch import CNNPyTorch
        from utils_vis import plot_history

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dl, val_dl, test_dl = get_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        model = CNNPyTorch().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        criterion = nn.CrossEntropyLoss()

        def evaluate(loader):
            model.eval()
            total_loss, correct = 0.0, 0
            y_true, y_pred = [], []
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    total_loss += loss.item() * x.size(0)
                    preds = logits.argmax(1)
                    correct += (preds == y).sum().item()
                    y_true.append(y.cpu())
                    y_pred.append(preds.cpu())
            y_true = torch.cat(y_true).numpy()
            y_pred = torch.cat(y_pred).numpy()
            return (
                total_loss / len(loader.dataset),
                correct / len(loader.dataset),
                precision_score(y_true, y_pred, average="macro"),
                recall_score(y_true, y_pred, average="macro"),
                f1_score(y_true, y_pred, average="macro"),
            )

        history: Dict[str, List[float]] = {}
        best_val_acc, patience, bad_epochs = 0.0, 5, 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            run_loss, run_correct = 0.0, 0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                run_loss += loss.item() * x.size(0)
                run_correct += (logits.argmax(1) == y).sum().item()

            scheduler.step()

            tr_loss = run_loss / len(train_dl.dataset)
            tr_acc = run_correct / len(train_dl.dataset)
            val_loss, val_acc, val_p, val_r, val_f1 = evaluate(val_dl)

            for k, v in [
                ("train_loss", tr_loss), ("val_loss", val_loss),
                ("train_acc", tr_acc), ("val_acc", val_acc),
                ("val_precision", val_p), ("val_recall", val_r),
                ("val_f1", val_f1)
            ]:
                history.setdefault(k, []).append(v)

            # Early-Stopping
            if val_acc > best_val_acc:
                best_val_acc, bad_epochs = val_acc, 0
                torch.save(model.state_dict(), out_dir / "best_model.pt")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"[Early Stop] epoch={epoch}")
                    break

            print(f"[{epoch:02d}/{args.epochs}] "
                  f"tr_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_acc={val_acc:.4%}")

        # ---- Test ----
        model.load_state_dict(torch.load(out_dir / "best_model.pt"))
        te_loss, te_acc, te_p, te_r, te_f1 = evaluate(test_dl)
        print(f"[Test] loss={te_loss:.4f} acc={te_acc:.4%} "
              f"P={te_p:.4%} R={te_r:.4%} F1={te_f1:.4%}")

        plot_history(history, out_dir / "metrics.png")

    # ========== TensorFlow ==========
    else:
        import numpy as np
        import tensorflow as tf
        from sklearn.metrics import precision_score, recall_score, f1_score

        from augment_tf import get_datasets
        from model_tf import build_tf_model
        from utils_vis import plot_history

        tf.keras.backend.clear_session()
        train_ds, val_ds, test_ds = get_datasets(
            batch_size=args.batch_size
        )

        model = build_tf_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        class ExtraMetrics(tf.keras.callbacks.Callback):
            def __init__(self):
                self.extra: Dict[str, List[float]] = {
                    "val_precision": [], "val_recall": [], "val_f1": []
                }

            def on_epoch_end(self, epoch, logs=None):
                y_true, y_pred = [], []
                for x, y in val_ds:
                    p = np.argmax(model(x, training=False), axis=1)
                    y_true.append(y.numpy())
                    y_pred.append(p)
                y_true = np.concatenate(y_true)
                y_pred = np.concatenate(y_pred)
                prec = precision_score(y_true, y_pred, average="macro")
                rec = recall_score(y_true, y_pred, average="macro")
                f1 = f1_score(y_true, y_pred, average="macro")
                for k, v in [("val_precision", prec),
                             ("val_recall", rec), ("val_f1", f1)]:
                    self.extra[k].append(v)
                print(f" - val_P:{prec:.4%} val_R:{rec:.4%} val_F1:{f1:.4%}")

        cb_extra = ExtraMetrics()
        cb_early = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        )
        hist = model.fit(
            train_ds,
            epochs=args.epochs,
            validation_data=val_ds,
            callbacks=[cb_extra, cb_early],
            verbose=2,
        )

        history = {k: v for k, v in hist.history.items()}
        history.update(cb_extra.extra)

        # ---- Test ----
        y_true, y_pred = [], []
        for x, y in test_ds:
            p = np.argmax(model(x, training=False), axis=1)
            y_true.append(y.numpy())
            y_pred.append(p)
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        te_loss, te_acc = model.evaluate(test_ds, verbose=0)
        te_p = precision_score(y_true, y_pred, average="macro")
        te_r = recall_score(y_true, y_pred, average="macro")
        te_f1 = f1_score(y_true, y_pred, average="macro")
        print(f"[Test] loss={te_loss:.4f} acc={te_acc:.4%} "
              f"P={te_p:.4%} R={te_r:.4%} F1={te_f1:.4%}")

        plot_history(history, out_dir / "metrics.png")


# ------- Windows / Spawn 安全入口 -------
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
