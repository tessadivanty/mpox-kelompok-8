import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim import Adam
from sklearn.metrics import f1_score
from Utils.getData import get_dataloaders, NUM_FOLDS
from Models.CNN import DoubleSimpleCNN

# KONFIGURASI
BATCH_SIZE = 32
EPOCH = 50
LEARNING_RATE = 0.0003
IMG_SIZE = 128
NUM_WORKERS = 4           # kalau error DataLoader di Windows, ganti 0
PATIENCE = 10              # early stopping: berhenti kalau valid F1 tidak membaik 10 epoch berturut-turut
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def run_epoch(model, loader, loss_function, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total = 0
    all_preds = []
    all_targets = []

    with torch.set_grad_enabled(training):
        for src, trg in loader:
            src = src.to(DEVICE)          # sudah (B, C, H, W), tidak perlu permute
            trg = trg.to(DEVICE)          # label berupa indeks kelas (0-5)
            pred = model(src)
            loss = loss_function(pred, trg)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()

            # Ambil kelas dengan probabilitas/logit tertinggi
            preds = torch.argmax(pred, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(trg.cpu().numpy())

    avg_loss = total / len(loader)
    # Gunakan average='macro' karena dataset memiliki bobot kelas berbeda (imbalanced)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return avg_loss, epoch_f1


def main():
    os.makedirs("Checkpoints", exist_ok=True)
    os.makedirs("Results", exist_ok=True)
    print("device:", DEVICE)

    for fold in range(1, NUM_FOLDS + 1):
        print(f"\n===== FOLD {fold}/{NUM_FOLDS} =====")

        # Train = Original Train + Augmented Train ; Valid hanya untuk validasi
        data = get_dataloaders(fold, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
                               num_workers=NUM_WORKERS, use_augmented=True, use_sampler=False)
        train_loader, valid_loader = data["train_loader"], data["valid_loader"]
        class_names = data["class_names"]
        print("kelas:", class_names, "| train/valid/test:", data["sizes"])

        model = DoubleSimpleCNN(input_dim=IMG_SIZE, input_c=3, output=len(class_names),
                                hidden_dim=128, dropout=0.5).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

        # kelas yang jumlah gambarnya sedikit diberi bobot lebih besar
        weights = data["class_weights"].to(DEVICE)
        print("bobot kelas:", dict(zip(class_names, [round(w, 3) for w in weights.tolist()])))
        loss_function = nn.CrossEntropyLoss(weight=weights)

        loss_train_all, loss_valid_all = [], []
        f1_train_all, f1_valid_all = [], []
        
        best_valid_f1 = -1.0  # Diubah dari float("inf") karena F1 makin tinggi makin bagus
        no_improve = 0
        
        for epoch in range(EPOCH):
            train_loss, train_f1 = run_epoch(model, train_loader, loss_function, optimizer)
            valid_loss, valid_f1 = run_epoch(model, valid_loader, loss_function)
            
            loss_train_all.append(train_loss)
            loss_valid_all.append(valid_loss)
            f1_train_all.append(train_f1)
            f1_valid_all.append(valid_f1)
            
            print(f"epoch = {epoch + 1} | train loss = {train_loss:.4f}, train F1 = {train_f1:.4f} | valid loss = {valid_loss:.4f}, valid F1 = {valid_f1:.4f}")

            if valid_f1 > best_valid_f1:      # simpan model terbaik berdasarkan Valid F1
                best_valid_f1 = valid_f1
                no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_loss,
                    'f1': valid_f1,
                    'classes': class_names,
                }, f"./Checkpoints/DoubleSimpleCNN_fold{fold}.pt")
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"Early stopping di epoch {epoch + 1} (F1 terbaik: {best_valid_f1:.4f} di epoch {epoch + 1 - PATIENCE})")
                    break

        # Plot Loss & F1 berdampingan
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        epochs_run = range(1, len(loss_train_all) + 1)
        
        ax1.plot(epochs_run, loss_train_all, color="#931a00", label='Training')
        ax1.plot(epochs_run, loss_valid_all, color="#3399e6", label='Validation')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"Loss - Fold {fold}")
        ax1.legend()

        ax2.plot(epochs_run, f1_train_all, color="#931a00", label='Training')
        ax2.plot(epochs_run, f1_valid_all, color="#3399e6", label='Validation')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Macro F1-Score")
        ax2.set_title(f"F1-Score - Fold {fold}")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f"./Results/training_fold{fold}.png")
        plt.close()

        with open(f"./Results/metrics_fold{fold}.json", "w") as f:
            json.dump({
                "train_loss": loss_train_all, 
                "valid_loss": loss_valid_all,
                "train_f1": f1_train_all,
                "valid_f1": f1_valid_all
            }, f)


if __name__ == "__main__":
    main()
