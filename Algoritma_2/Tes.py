
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from Models.CNN import SimpleCNN                         # noqa: E402
from Utils.getdata import NUM_FOLDS, get_test_loader     # noqa: E402

MODEL_DIR = ROOT / "Models"
RESULT_DIR = ROOT / "Results"


@torch.no_grad()
def predict(model, loader, device, tta=False):
    """Mengembalikan (y_true, probabilitas softmax)."""
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        prob = torch.softmax(model(x), dim=1)
        if tta:
            prob = (prob + torch.softmax(model(torch.flip(x, dims=[3])), dim=1)) / 2
        ps.append(prob.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps).astype(np.float64)


def compute_metrics(y_true, probs, num_classes):
    y_pred = probs.argmax(1)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    try:
        p_norm = probs / probs.sum(1, keepdims=True)
        auc = roc_auc_score(y_true, p_norm, multi_class="ovr", average="macro",
                            labels=list(range(num_classes)))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": p_m, "recall_macro": r_m, "f1_macro": f_m,
        "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f_w,
        "roc_auc_ovr_macro": auc,
    }


def plot_confusion(cm, class_names, path, title, normalize=False):
    data = cm.astype(float)
    if normalize:
        data = data / data.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label sebenarnya")
    ax.set_title(title)
    thr = data.max() / 2
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            txt = f"{data[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if data[i, j] > thr else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def plot_roc(y_true, probs, class_names, path, title):
    n = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n)))
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_bin[:, i], probs[:, i]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--tta", action="store_true", help="test-time augmentation (flip horizontal)")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(1, NUM_FOLDS + 1)))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULT_DIR.mkdir(exist_ok=True)
    print(f"Device: {device} | TTA: {args.tta}")

    all_metrics, all_y, all_p = {}, [], []
    cm_total, class_names = None, None

    for fold in args.folds:
        ckpt_path = MODEL_DIR / f"cnn_fold{fold}.pth"
        if not ckpt_path.exists():
            print(f"[SKIP] {ckpt_path} tidak ada. Jalankan Train.py dulu.")
            continue

        ckpt = torch.load(ckpt_path, map_location=device)
        class_names = ckpt["class_names"]
        model = SimpleCNN(num_classes=len(class_names), dropout=ckpt.get("dropout", 0.4)).to(device)
        model.load_state_dict(ckpt["model_state"])

        loader, _ = get_test_loader(fold, img_size=ckpt["img_size"],
                                    batch_size=args.batch_size, num_workers=args.num_workers)
        y_true, probs = predict(model, loader, device, tta=args.tta)
        y_pred = probs.argmax(1)

        m = compute_metrics(y_true, probs, len(class_names))
        all_metrics[fold] = m
        all_y.append(y_true)
        all_p.append(probs)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        cm_total = cm if cm_total is None else cm_total + cm
        plot_confusion(cm, class_names, RESULT_DIR / f"confusion_fold{fold}.png",
                       f"Confusion Matrix - Fold {fold} (Test)")

        report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
        (RESULT_DIR / f"report_fold{fold}.txt").write_text(report)

        print(f"\n--- Fold {fold} (n_test={len(y_true)}) ---")
        print(f"Acc {m['accuracy']:.4f} | Prec {m['precision_macro']:.4f} | Rec {m['recall_macro']:.4f} | "
              f"F1 {m['f1_macro']:.4f} | AUC {m['roc_auc_ovr_macro']:.4f}")
        print(report)

    if not all_metrics:
        print("Tidak ada model yang dievaluasi.")
        return

    # Rata-rata 5-fold 
    keys = list(next(iter(all_metrics.values())).keys())
    mean = {k: float(np.nanmean([m[k] for m in all_metrics.values()])) for k in keys}
    std = {k: float(np.nanstd([m[k] for m in all_metrics.values()])) for k in keys}

    print(f"\n{'=' * 70}\nRATA-RATA {len(all_metrics)}-FOLD (Test)\n{'=' * 70}")
    for k in keys:
        print(f"{k:22s}: {mean[k]:.4f} +/- {std[k]:.4f}")

    with open(RESULT_DIR / "summary_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold"] + keys)
        for fold, m in all_metrics.items():
            w.writerow([fold] + [f"{m[k]:.4f}" for k in keys])
        w.writerow(["mean"] + [f"{mean[k]:.4f}" for k in keys])
        w.writerow(["std"] + [f"{std[k]:.4f}" for k in keys])
    with open(RESULT_DIR / "summary_metrics.json", "w") as f:
        json.dump({"per_fold": all_metrics, "mean": mean, "std": std}, f, indent=2)

    # Confusion matrix gabungan semua fold + ROC (data gabungan semua fold)
    plot_confusion(cm_total, class_names, RESULT_DIR / "confusion_total.png",
                   "Confusion Matrix - Total semua fold (Test)")
    plot_confusion(cm_total, class_names, RESULT_DIR / "confusion_total_normalized.png",
                   "Confusion Matrix (dinormalisasi) - Total semua fold", normalize=True)
    plot_roc(np.concatenate(all_y), np.concatenate(all_p), class_names,
             RESULT_DIR / "roc_curve_total.png", "ROC Curve (OvR) - gabungan semua fold")
    print(f"\nHasil tersimpan di: {RESULT_DIR}")


if __name__ == "__main__":
    main()
