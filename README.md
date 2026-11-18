# Klasifikasi Citra Lesi Kulit Monkeypox (6 Kelas) Menggunakan Perbandingan Arsitektur CNN dan EfficientNetV2

Proyek ini mengimplementasikan dan membandingkan performa tiga arsitektur *Deep Learning* berbasis PyTorch untuk klasifikasi multi-kelas citra lesi kulit (**6 kelas**: *Chickenpox*, *Cowpox*, *HFMD*, *Healthy*, *Measles*, dan *Monkeypox*) menggunakan skema evaluasi **5-Fold Cross-Validation**.

---

## Ringkasan Eksperimen & Model

Seluruh model menggunakan pipeline prapemrosesan dan partisi *5-fold cross-validation* yang seragam (`Utils/getData.py`), dengan tiga variasi arsitektur model:

1. **Algoritma 1 — `DoubleSimpleCNN` (Custom 2-Stage CNN)**
   * **Arsitektur:** Dua *stage* konvolusi berurutan dengan kernel $5 \times 5$ (`3 -> 20 -> 50 -> 100 -> 100` kanal), `BatchNorm2d`, `ReLU`, `MaxPool2d(2)`, serta *classifier* `Linear(6400 -> 128 -> 6)` dengan `Dropout(0.5)` (~1,22 juta parameter).
   * **Konfigurasi:** Input $128 \times 128$, optimizer `Adam` (`lr=3e-4`, `weight_decay=1e-4`), weighted `CrossEntropyLoss`, dan *early stopping* berdasarkan `val_loss`.
2. **Algoritma 2 — `SimpleCNN` 4-Blok (Modified Deep Custom CNN)**
   * **Arsitektur:** Empat blok konvolusi ganda dengan kernel $3 \times 3$ (`3 -> 32 -> 64 -> 128 -> 256` kanal), `BatchNorm2d`, `ReLU`, `MaxPool2d(2)`, `Dropout2d` bertingkat (`0.05`–`0.20`), `AdaptiveAvgPool2d(1)` (*Global Average Pooling*), serta *classifier* `Linear(256 -> 128 -> 6)` (~1,21 juta parameter).
   * **Konfigurasi:** Input $128 \times 128$, optimizer `AdamW` (`lr=1e-3`), `ReduceLROnPlateau`, *label smoothing* (`0.05`), dan *early stopping* berdasarkan `val_f1` (*macro*).
3. **Algoritma 3 — `EfficientNetV2-S` (*Transfer Learning*)**
   * **Arsitektur:** Backbone `efficientnet_v2_s` dengan bobot *pretrained* `IMAGENET1K_V1` dan modifikasi lapisan akhir `classifier[1]` menjadi `Linear(1280 -> 6)` (~20,2 juta parameter).
   * **Konfigurasi:** Input $224 \times 224$, optimizer `AdamW` (`lr=3e-4`), `CosineAnnealingLR`, weighted `CrossEntropyLoss`, *mixed precision* (AMP), dan *early stopping* berdasarkan `val_f1` (*macro*).

---

## Dataset & Pipeline Preprocessing

### 1. Distribusi Kelas Data Uji (Total 5-Fold = 364 Citra Asli)
Partisi **Train** menggabungkan direktori `Original Images` dan `Augmented Images`, sedangkan partisi **Valid** dan **Test** menggunakan citra murni dari `Original Images` untuk mencegah *data leakage*.

| Kelas | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Total Uji | Proporsi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Monkeypox** | 27 | 31 | 22 | 27 | 18 | **125** | 34,34% |
| **HFMD** | 16 | 20 | 16 | 15 | 15 | **82** | 22,53% |
| **Healthy** | 9 | 11 | 11 | 9 | 10 | **50** | 13,74% |
| **Chickenpox** | 9 | 11 | 6 | 9 | 7 | **42** | 11,54% |
| **Cowpox** | 7 | 4 | 4 | 13 | 7 | **35** | 9,62% |
| **Measles** | 6 | 5 | 5 | 8 | 6 | **30** | 8,24% |
| **Total** | **74** | **82** | **64** | **81** | **63** | **364** | **100,00%** |

### 2. Tahapan Preprocessing (`Utils/getData.py`)
* **Konversi & Resize:** Seluruh gambar dikonversi ke RGB dan diubah ukurannya ke $128 \times 128$ (Model 1 & 2) atau $224 \times 224$ (Model 3).
* **Augmentasi On-the-Fly (khusus Train):** `RandomHorizontalFlip(0.5)`, `RandomVerticalFlip(0.2)`, `RandomRotation(15)`, dan `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02)`.
* **Normalisasi:** Rentang piksel $[-1, 1]$ dengan `MEAN = (0.5, 0.5, 0.5)` dan `STD = (0.5, 0.5, 0.5)`.
* **Penanganan Imbalance:** Pembobotan kelas otomatis berbasis *inverse class frequency* pada `CrossEntropyLoss`.

---

## Hasil Evaluasi 5-Fold Cross-Validation

Tabel berikut merangkum performa rata-rata ($\text{Mean} \pm \text{Std}$) pada set pengujian (*Test Set*) di seluruh 5 *fold*:

| Model | Resolusi | Accuracy | Precision | Recall | F1-Score | ROC-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Algoritma 1: `DoubleSimpleCNN`** | $128 \times 128$ | $0.4905 \pm 0.0928$ | $0.5449 \pm 0.0548^*$ | $0.4905 \pm 0.0928^*$ | $0.4880 \pm 0.0853^*$ | $0.7982 \pm 0.0508$ |
| **Algoritma 2: `SimpleCNN` (4-Blok)** | $128 \times 128$ | $0.6919 \pm 0.0963$ | $0.7010 \pm 0.1278$ | $0.6830 \pm 0.1099$ | $0.6783 \pm 0.1138$ | $0.9020 \pm 0.0375$ |
| **Algoritma 3: `EfficientNetV2-S`** | $224 \times 224$ | **$0.7756 \pm 0.0403$** | **$0.7820 \pm 0.0739$** | **$0.7747 \pm 0.0417$** | **$0.7635 \pm 0.0539$** | **$0.9475 \pm 0.0072$** |

> *Catatan: Metrik Precision, Recall, dan F1-Score pada Algoritma 1 menggunakan `weighted average`, sedangkan Algoritma 2 dan Algoritma 3 menggunakan `macro average` (pada Algoritma 2, nilai `weighted F1` adalah $0.6935 \pm 0.0944$).*

### Kesimpulan Utama
* **Transfer Learning Unggul Signifikan:** `EfficientNetV2-S` mencapai akurasi rata-rata tertinggi (**77,56%**) dan ROC-AUC **0,9475** dengan stabilitas antar-fold terbaik ($\pm 4,03\%$).
* **Dampak Desain Arsitektur CNN Kustom:** Penambahan kedalaman menjadi 4 blok konvolusi ganda $3 \times 3$ disertai *Global Average Pooling* dan *label smoothing* pada Algoritma 2 meningkatkan akurasi sebesar **+20,14%** dibandingkan `DoubleSimpleCNN`.
* **Tantangan Klasifikasi:** Kesalahan klasifikasi paling sering terjadi pada pasangan lesi vesikular yang memiliki kemiripan morfologi visual tinggi, yaitu antara **Monkeypox**, **Chickenpox**, dan **Cowpox**.

---

## Struktur Direktori Repositori

```text
├── Dataset/
│   ├── Original Images/         # Citra asli per Fold (Fold1..Fold5 -> Train/Valid/Test)
│   └── Augmented Images/        # Citra augmentasi luring per Fold (Train)
├── Utils/
│   ├── getData.py               # Dataset loader, transformasi augmentasi, dan class weights
│   └── metrics.py               # Fungsi perhitungan metrik & visualisasi (Algoritma 3)
├── Algoritma_1/
│   ├── Models/CNN.py            # Arsitektur DoubleSimpleCNN
│   ├── train.py                 # Skrip training 5-fold Algoritma 1
│   ├── test.py                  # Skrip evaluasi & confusion matrix Algoritma 1
│   └── Results/                 # Metrik CSV, kurva loss, dan confusion matrix
├── Algoritma_2/
│   ├── Models/CNN.py            # Arsitektur SimpleCNN 4-Blok + GAP
│   ├── Train.py                 # Skrip training 5-fold Algoritma 2
│   ├── Tes.py                   # Skrip evaluasi, classification report, & ROC curve
│   └── Results/                 # Summary metrics, report per fold, dan visualisasi
├── Algoritma_3/
│   ├── Models/EfficientNetV2.py # Builder model EfficientNetV2-S pretrained
│   ├── traintes.py              # Skrip training & testing 5-fold Algoritma 3
│   └── results/efficientnetv2/  # Metrik JSON/CSV, kurva training, dan confusion matrix
├── requirements.txt
└── README.md
```

---

## Instalasi & Cara Menjalankan

### 1. Instalasi Dependensi
Pastikan Python $\ge 3.9$ sudah terpasang, lalu instal pustaka yang dibutuhkan:
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn pillow tabulate
```

### 2. Verifikasi Dataset
Pastikan folder `Dataset/Original Images` dan `Dataset/Augmented Images` berada di direktori utama proyek, lalu jalankan:
```bash
python Utils/getData.py
```

### 3. Menjalankan Training & Evaluasi Tiap Algoritma

* **Algoritma 1 (`DoubleSimpleCNN`):**
  ```bash
  python Algoritma_1/train.py
  python Algoritma_1/test.py
  ```

* **Algoritma 2 (`SimpleCNN` 4-Blok):**
  ```bash
  python Algoritma_2/Train.py --epochs 50 --batch_size 32 --img_size 128 --lr 1e-3
  python Algoritma_2/Tes.py
  ```

* **Algoritma 3 (`EfficientNetV2-S`):**
  ```bash
  python Algoritma_3/traintes.py --epochs 50 --batch_size 32 --img_size 224 --lr 3e-4 --patience 10
  ```
