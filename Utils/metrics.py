import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                            recall_score, roc_auc_score, roc_curve)


#prediksi & metrik
@torch.no_grad()
def predict(model, loader, dev):
    """Return (probabilitas, label asli, rata-rata loss)."""
    model.eval()
    probs, ys, loss_sum, n = [], [], 0.0, 0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            out = model(x)
        loss_sum += ce(out.float(), y).item()
        n += len(y)
        probs.append(torch.softmax(out.float(), 1).cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(ys), loss_sum / n


def compute_metrics(y, p, n_cls):
    """Accuracy, precision, recall, F1, ROC-AUC + confusion matrix."""
    pred = p.argmax(1)
    if n_cls == 2:
        avg, auc = "binary", roc_auc_score(y, p[:, 1])
    else:
        avg = "macro"
        auc = roc_auc_score(y, p, multi_class="ovr", average="macro", labels=list(range(n_cls)))
    m = dict(accuracy=accuracy_score(y, pred),
                precision=precision_score(y, pred, average=avg, zero_division=0),
                recall=recall_score(y, pred, average=avg, zero_division=0),
                f1=f1_score(y, pred, average=avg, zero_division=0),
                roc_auc=auc)
    return m, confusion_matrix(y, pred, labels=list(range(n_cls)))


#plot
def plot_confusion_matrix(cm, names, path, title):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names,
            xlabel="Prediksi", ylabel="Aktual", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(names)):
        for j in range(len(names)):
            txt = f"{cm[i, j]:.0f}" if float(cm[i, j]).is_integer() else f"{cm[i, j]:.1f}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(h, path, title):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(h["train_loss"], label="train")
    ax[0].plot(h["val_loss"], label="valid")
    ax[0].set(title=f"{title} - Loss", xlabel="epoch")
    ax[0].legend()
    ax[1].plot(h["val_f1"], label="valid F1")
    ax[1].plot(h["val_acc"], label="valid acc")
    ax[1].set(title=f"{title} - Validasi", xlabel="epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y, p_pos, auc, path, title):
    fpr, tpr, _ = roc_curve(y, p_pos)
    plt.figure(figsize=(4.5, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", c="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()