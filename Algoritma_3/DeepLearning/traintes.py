import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from Models.EfficientNetV2 import build_efficientnetv2
from Utils.getdata import NUM_FOLDS, get_dataloaders
from Utils.metrics import compute_metrics, plot_confusion_matrix, plot_history, predict


def run_fold(k, args, dev, out):
    # 1. data
    data = get_dataloaders(k, img_size=args.img_size, batch_size=args.batch_size,
                            num_workers=args.workers, use_augmented=True)
    ltr, lva, lte = data["train_loader"], data["valid_loader"], data["test_loader"]
    names, weights = data["class_names"], data["class_weights"]
    n_cls = len(names)
    print(f"[fold {k}] train/valid/test = {data['sizes']} kelas={names}")

    # 2. model, loss, optimizer
    model = build_efficientnetv2(n_cls).to(dev)
    criterion = nn.CrossEntropyLoss(weight=weights.to(dev))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    # 3. training loop + early stopping
    hist = {key: [] for key in ["train_loss", "val_loss", "val_acc", "val_f1"]}
    best_f1, best_state, best_ep, bad = -1.0, None, 0, 0

    for ep in range(args.epochs):
        model.train()
        total_loss, n = 0.0, 0
        for i, (x, y) in enumerate(ltr):
            x, y = x.to(dev), y.to(dev)
            if i % 50 == 0:
                print(f"    batch {i}/{len(ltr)}", flush=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = criterion(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(y)
            n += len(y)
        scheduler.step()

        # validasi
        p, y, val_loss = predict(model, lva, dev)
        m, _ = compute_metrics(y, p, n_cls)
        hist["train_loss"].append(total_loss / n)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(m["accuracy"])
        hist["val_f1"].append(m["f1"])
        print(f"  ep {ep+1:02d}/{args.epochs} train_loss={total_loss/n:.4f} "
                f"val_loss={val_loss:.4f} val_f1={m['f1']:.4f}")

        # early stopping: F1 validasi harus naik minimal min_delta (toleransi)
        if m["f1"] > best_f1 + args.min_delta:
            best_f1, best_ep, bad = m["f1"], ep + 1, 0
            best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  EARLY STOP di epoch {ep+1} (terbaik: epoch {best_ep}, val_f1={best_f1:.4f})")
                break

    # 4. testing dengan bobot epoch terbaik (hanya di memori, tidak disimpan ke file)
    model.load_state_dict(best_state)
    p, y, _ = predict(model, lte, dev)
    m, cm = compute_metrics(y, p, n_cls)
    m = {kk: float(v) for kk, v in m.items()}
    m["best_epoch"] = best_ep
    m["epochs_run"] = len(hist["train_loss"])

    plot_confusion_matrix(cm, names, out / f"fold{k}_confusion_matrix.png", f"Fold {k} - Test")
    plot_history(hist, out / f"fold{k}_training_curve.png", f"Fold {k}")
    with open(out / f"fold{k}_metrics.json", "w") as f:
        json.dump(dict(fold=k, metrics=m, confusion_matrix=cm.tolist(), classes=names, history=hist), f, indent=2)

    print(f"[fold {k}] TEST", {kk: round(v, 4) for kk, v in m.items()})
    return m, cm, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=50, help="batas maksimal epoch")
    ap.add_argument("--patience", type=int, default=10,
                    help="early stopping: berhenti jika F1 valid tidak membaik selama N epoch")
    ap.add_argument("--min_delta", type=float, default=0.0,
                    help="toleransi: kenaikan F1 valid minimal agar dianggap membaik")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(1, NUM_FOLDS + 1)))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir) / "efficientnetv2"
    out.mkdir(parents=True, exist_ok=True)
    print("device:", dev, "| output:", out)

    rows, cms, names = [], [], None
    t0 = time.time()
    for k in args.folds:
        m, cm, names = run_fold(k, args, dev, out)
        rows.append(dict(fold=k, **m))
        cms.append(cm)

    # rerata 5-fold
    df = pd.DataFrame(rows)
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    mean, std = df[cols].mean(), df[cols].std(ddof=1)
    df.loc[len(df)] = dict(fold="mean", **mean.to_dict())
    df.loc[len(df)] = dict(fold="std", **std.to_dict())
    df.round(4).to_csv(out / "results_5fold.csv", index=False)
    plot_confusion_matrix(np.mean(cms, 0), names, out / "confusion_matrix_mean.png", "Rerata 5-fold - Test")
    (out / "summary.md").write_text(df.round(4).to_markdown(index=False))

    print("\n=== Rerata 5-fold ===\n", df.round(4).to_string(index=False))
    print(f"selesai dalam {(time.time()-t0)/60:.1f} menit")


if __name__ == "__main__":
    main()