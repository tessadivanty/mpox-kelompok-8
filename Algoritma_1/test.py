import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sn

from Utils.getData import get_test_loader, NUM_FOLDS
from Models.CNN import DoubleSimpleCNN
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)

# ====== KONFIGURASI (harus sama dengan train.py) ======
BATCH_SIZE = 32
IMG_SIZE = 128
NUM_WORKERS = 4           # kalau error DataLoader di Windows, ganti 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    rows, cm_sum, classes = [], None, None

    for fold in range(1, NUM_FOLDS + 1):
        checkpoint = torch.load(f"./Checkpoints/DoubleSimpleCNN_fold{fold}.pt",
                                map_location=DEVICE, weights_only=False)
        classes = checkpoint['classes']

        test_loader, _ = get_test_loader(fold, IMG_SIZE, BATCH_SIZE, NUM_WORKERS)

        model = DoubleSimpleCNN(input_dim=IMG_SIZE, input_c=3, output=len(classes),
                                hidden_dim=128, dropout=0.5).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])

        prediction, ground_truth, probs = [], [], []
        with torch.no_grad():
            model.eval()
            for src, trg in test_loader:
                src = src.to(DEVICE)
                pred = torch.softmax(model(src), dim=1).cpu().numpy()   # probabilitas untuk ROC-AUC
                probs.extend(pred)
                prediction.extend(np.argmax(pred, axis=1))
                ground_truth.extend(trg.numpy())                        # label sudah berupa indeks

        probs = np.array(probs)
        try:
            auc = roc_auc_score(ground_truth, probs, multi_class='ovr', average='macro',
                                labels=list(range(len(classes))))
        except ValueError:      # ada kelas yang tidak muncul di test fold ini
            auc = np.nan

        row = {
            "fold": fold,
            "accuracy": accuracy_score(ground_truth, prediction),
            "precision": precision_score(ground_truth, prediction, average='macro', zero_division=0),
            "recall": recall_score(ground_truth, prediction, average='macro', zero_division=0),
            "f1": f1_score(ground_truth, prediction, average='macro', zero_division=0),
            "roc_auc": auc,
        }
        rows.append(row)
        print(row)

        cm = confusion_matrix(ground_truth, prediction, labels=list(range(len(classes))))
        cm_sum = cm if cm_sum is None else cm_sum + cm

    df = pd.DataFrame(rows)
    mean, std = df.drop(columns="fold").mean(), df.drop(columns="fold").std()
    print("\n===== RERATA 5-FOLD =====")
    for m in mean.index:
        print(f"{m:10s}: {mean[m]:.4f} (+/- {std[m]:.4f})")

    df.loc[len(df)] = ["mean"] + mean.tolist()
    df.to_csv("./Results/metrics.csv", index=False)

    # confusion matrix gabungan seluruh fold (dinormalisasi per baris)
    df_cm = pd.DataFrame(cm_sum / cm_sum.sum(axis=1, keepdims=True), index=classes, columns=classes)
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues")
    plt.xlabel("Prediksi")
    plt.ylabel("Sebenarnya")
    plt.savefig("./Results/confusion_matrix.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
