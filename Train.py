
import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from Models.CNN import SimpleCNN                    # noqa: E402
from Utils.getdata import NUM_FOLDS, get_dataloaders  # noqa: E402

MODEL_DIR = ROOT / "Models"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False):
    """Satu epoch. Jika optimizer diberikan -> mode training, selain itu evaluasi."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    all_pred, all_true = [], []

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            n += x.size(0)
            all_pred.append(pred.cpu())
            all_true.append(y.cpu())

    f1 = f1_score(torch.cat(all_true).numpy(), torch.cat(all_pred).numpy(),
                  average="macro", zero_division=0)
    return total_loss / n, correct / n, f1


def plot_history(hist, path, fold):
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, hist["train_loss"], label="train")
    ax[0].plot(ep, hist["val_loss"], label="valid")
    ax[0].set_title(f"Loss - Fold {fold}")
    ax[0].set_xlabel("Epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ep, hist["train_acc"], label="train")
    ax[1].plot(ep, hist["val_acc"], label="valid")
    ax[1].set_title(f"Accuracy - Fold {fold}")
    ax[1].set_xlabel("Epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def train_one_fold(fold, args, device):
    print(f"\n{'=' * 70}\nFOLD {fold}/{NUM_FOLDS}\n{'=' * 70}")
    data = get_dataloaders(fold, img_size=args.img_size, batch_size=args.batch_size,
                           num_workers=args.num_workers, use_augmented=True,
                           use_sampler=args.use_sampler)
    class_names = data["class_names"]
    ntr, nva, nte = data["sizes"]
    print(f"Kelas : {class_names}")
    print(f"Data  : train={ntr} (original+augmented) | valid={nva} | test={nte}")

    set_seed(args.seed + fold)
    model = SimpleCNN(num_classes=len(class_names), dropout=args.dropout).to(device)

    # Loss: CrossEntropy berbobot kelas + label smoothing (training),
    #       CrossEntropy biasa untuk validasi.
    # Jika sampler dipakai, batch sudah seimbang -> bobot kelas di loss dimatikan
    # agar ketidakseimbangan tidak dikoreksi dua kali.
    cw = None if args.use_sampler else data["class_weights"].to(device)
    train_criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
    val_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)

    use_amp = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr"]}
    best = {"f1": -1.0, "acc": -1.0, "loss": float("inf"), "epoch": 0}
    wait = 0
    ckpt_path = MODEL_DIR / f"cnn_fold{fold}.pth"
    MODEL_DIR.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, _ = run_epoch(model, data["train_loader"], train_criterion, device,
                                       optimizer, scaler, use_amp)
        va_loss, va_acc, va_f1 = run_epoch(model, data["valid_loader"], val_criterion, device,
                                           use_amp=use_amp)
        scheduler.step(va_loss)
        lr = optimizer.param_groups[0]["lr"]

        for k, v in zip(hist.keys(), [tr_loss, tr_acc, va_loss, va_acc, va_f1, lr]):
            hist[k].append(v)

        # Model terbaik dipilih berdasarkan macro-F1 validasi (adil untuk data tidak seimbang);
        # jika macro-F1 sama, pilih val loss yang lebih rendah.
        improved = (va_f1 > best["f1"] + 1e-4) or \
                   (abs(va_f1 - best["f1"]) <= 1e-4 and va_loss < best["loss"])
        if improved:
            best.update(f1=va_f1, acc=va_acc, loss=va_loss, epoch=epoch)
            wait = 0
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names,
                "img_size": args.img_size,
                "dropout": args.dropout,
                "fold": fold,
                "epoch": epoch,
                "val_acc": float(va_acc),
                "val_f1": float(va_f1),
                "val_loss": float(va_loss),
            }, ckpt_path)
            mark = " *"
        else:
            wait += 1
            mark = ""

        print(f"Ep {epoch:03d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f} | lr {lr:.1e} | "
              f"{time.time() - t0:.0f}s{mark}")

        if wait >= args.patience:
            print(f"Early stopping (tidak membaik {args.patience} epoch).")
            break

    print(f"Terbaik fold {fold}: epoch {best['epoch']} | val F1 {best['f1']:.4f} | "
          f"val acc {best['acc']:.4f} | val loss {best['loss']:.4f}")

    with open(MODEL_DIR / f"history_fold{fold}.json", "w") as f:
        json.dump(hist, f, indent=2)
    plot_history(hist, MODEL_DIR / f"curve_fold{fold}.png", fold)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--use_sampler", action="store_true",
                    help="pakai WeightedRandomSampler (bobot kelas di loss dimatikan)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(1, NUM_FOLDS + 1)))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    results = {}
    for fold in args.folds:
        results[fold] = train_one_fold(fold, args, device)

    accs = [r["acc"] for r in results.values()]
    f1s = [r["f1"] for r in results.values()]
    print(f"\n{'=' * 70}\nRINGKASAN VALIDASI")
    for k, r in results.items():
        print(f"  Fold {k}: val F1 {r['f1']:.4f} | val acc {r['acc']:.4f} (epoch {r['epoch']})")
    print(f"  Rata-rata val F1 : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"  Rata-rata val acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print("Lanjutkan dengan:  python Tes.py")


if __name__ == "__main__":      # wajib di Windows karena DataLoader multiprocessing
    main()
