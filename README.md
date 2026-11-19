# Klasifikasi Citra Penyakit Kulit Mpox dan Ruam Serupa Menggunakan Convolutional Neural Network (CNN) — Kelompok 8

Repositori ini berisi kode implementasi, skenario pelatihan, dan hasil evaluasi komparatif tiga arsitektur *Deep Learning* berbasis *Convolutional Neural Network* (CNN) untuk klasifikasi enam kelas citra penyakit kulit (**Chickenpox, Cowpox, HFMD, Healthy, Measles, dan Monkeypox**) menggunakan protokol *5-Fold Cross-Validation*.

---

## 🎯 Gambaran Umum

Deteksi dini penyakit **Mpox (*Monkeypox*)** secara visual sering menghadapi kendala karena kemiripan morfologi lesi kulit dengan penyakit ruam lainnya. Untuk membandingkan performa tiga arsitektur model secara adil (*apple-to-apple*) pada kondisi distribusi data yang tidak seimbang (*class imbalance*), eksperimen ini menerapkan:

1. **Resolusi Masukan Seragam:** Seluruh citra masukan untuk ketiga algoritma diseragamkan ke ukuran $128 \times 128$ piksel.
2. **Penanganan Ketidakseimbangan Kelas:** Menggunakan pembobotan kelas (*class weighting*) pada fungsi **Weighted Cross-Entropy Loss** berdasarkan frekuensi sampel tiap kelas pada data pelatihan.
3. **Early Stopping Berbasis Validasi Macro F1:** Pelatihan dihentikan secara otomatis apabila nilai *validation macro F1-score* tidak membaik selama **10 epoch berturut-turut** (*patience* $= 10$) guna memaksimalkan performa pada kelas minoritas.
4. **Evaluasi Metrik Macro:** Seluruh metrik evaluasi dihitung menggunakan rata-rata makro (*macro-averaging*) di atas skema **5-Fold Cross-Validation**.

---

## 📊 Karakteristik Dataset

Dataset mencakup 6 kategori klinis dengan total **364 citra pengujian** yang terdistribusi pada lima lipatan (*fold*):

| Kelas | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Total Uji | Proporsi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Chickenpox** | 9 | 11 | 6 | 9 | 7 | 42 | 11,54% |
| **Cowpox** | 7 | 4 | 4 | 13 | 7 | 35 | 9,61% |
| **HFMD** | 16 | 20 | 16 | 15 | 15 | 82 | 22,53% |
| **Healthy** | 9 | 11 | 11 | 9 | 10 | 50 | 13,74% |
| **Measles** | 6 | 5 | 5 | 8 | 6 | 30 | 8,24% |
| **Monkeypox** | 27 | 31 | 22 | 27 | 18 | 125 | 34,34% |
| **Total** | **74** | **82** | **64** | **81** | **63** | **364** | **100,00%** |

---

## ⚙️ Metodologi & Protokol Eksperimen

### 1. Pra-pemrosesan & Augmentasi (`Utils/getData.py`)

* **Data Pelatihan (`Train`):** Menggabungkan `Original Images` dan `Augmented Images` (`use_augmented=True`), ditambah augmentasi *on-the-fly*:
  * `Resize((128, 128))`
  * `RandomHorizontalFlip(0.5)` & `RandomVerticalFlip(0.2)`
  * `RandomRotation(15)`
  * `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02)`
  * `ToTensor()` & `Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))`
* **Data Validasi & Pengujian (`Valid` & `Test`):** Hanya melalui proses `Resize((128, 128))`, `ToTensor()`, dan `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))` tanpa transformasi acak.

### 2. Pembobotan Kelas & Kriteria Berhenti

* **Bobot Kelas:** $w_c = \frac{N}{C \times N_c}$ diterapkan pada `nn.CrossEntropyLoss(weight=weights)` untuk menyeimbangkan penalti gradien antara kelas mayoritas dan minoritas.
* **Early Stopping:** Memantau peningkatan `valid_f1` (*macro F1-score*) dengan `patience = 10` dan batas maksimal `50` *epoch*.

---

## 🧠 Detail Arsitektur Model

| Aspek | Algoritma 1 (`DoubleSimpleCNN`) | Algoritma 2 (`SimpleCNN` 4-Blok) | Algoritma 3 (`EfficientNetV2-S`) |
| :--- | :--- | :--- | :--- |
| **Pendekatan** | CNN Kustom 2-Stage | CNN Kustom 4-Blok Konvolusi Ganda | *Transfer Learning* (*ImageNet-1K*) |
| **Struktur Ekstraktor Fitur** | 2 Stage $\times$ 2 Blok (`Conv2d 5×5` $\rightarrow$ `BN` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool 2×2`), kanal: `20 -> 50 -> 100 -> 100` | 4 Blok (`[Conv2d 3×3 -> BN -> ReLU] × 2 -> MaxPool 2×2 -> Dropout2d`), kanal: `32 -> 64 -> 128 -> 256` | Backbone `efficientnet_v2_s` (*Fused-MBConv* & *MBConv* + *Squeeze-and-Excitation*) |
| **Pooling & Classifier** | `Flatten(6400)` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Linear(128)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Linear(6)` | `AdaptiveAvgPool2d(1)` $\rightarrow$ `Dropout(0.4)` $\rightarrow$ `Linear(128)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.4)` $\rightarrow$ `Linear(6)` | Modifikasi `classifier[1]` menjadi `Linear(1280, 6)` |
| **Optimizer & LR** | `Adam` (`lr=3e-4`, `weight_decay=1e-4`) | `AdamW` (`lr=1e-3`, `weight_decay=1e-4`) + `ReduceLROnPlateau` + *Label Smoothing* (`0.05`) | `AdamW` (`lr=3e-4`, `weight_decay=1e-4`) + `CosineAnnealingLR` (`T_max=50`) |
| **Resolusi Input** | $128 \times 128$ | $128 \times 128$ | $128 \times 128$ (`--img_size 128`) |

---

## 📈 Ringkasan Hasil Evaluasi

Rata-rata performa pengujian pada data uji di seluruh **5-Fold Cross-Validation** menggunakan metrik **Macro**:

| Model | Akurasi | Presisi (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Algoritma 1 (`DoubleSimpleCNN`)** | 0,6302 | 0,6585 | 0,5998 | 0,6015 | 0,8765 |
| **Algoritma 2 (`SimpleCNN` 4-Blok Ganda)** | 0,6919 | 0,7010 | 0,6830 | 0,6783 | 0,9020 |
| **Algoritma 3 (`EfficientNetV2-S`)** | **0,7709** | **0,7523** | **0,7672** | **0,7491** | **0,9444** |

> **Kesimpulan Utama:** Arsitektur *transfer learning* **EfficientNetV2-S** (Algoritma 3) memberikan performa tertinggi dan paling konsisten di seluruh lipatan pengujian (mencapai puncak akurasi **0,8889** dan *macro F1-score* **0,8657** pada Fold 5), membuktikan efektivitas kombinasi bobot pra-latih *ImageNet-1K*, *Weighted Cross-Entropy Loss*, dan *early stopping* berbasis *valid macro F1*.

---

## 🚀 Cara Instalasi & Menjalankan Proyek

1. **Clone repositori ini:**
   ```bash
   git clone https://github.com/tessadivanty/mpox-kelompok-8.git
   cd mpox-kelompok-8
   ```

2. **Instal seluruh dependensi dari `requirements.txt`:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Siapkan direktori dataset:**
   Pastikan folder `Dataset/Original Images` dan `Dataset/Augmented Images` memiliki struktur sub-folder `Fold 1` hingga `Fold 5` dengan pembagian `Train`, `Valid`, dan `Test`.

4. **Jalankan pelatihan dan pengujian masing-masing algoritma:**
   * **Algoritma 1 (`DoubleSimpleCNN`):**
     ```bash
     python train.py
     python test.py
     ```
   * **Algoritma 2 (`SimpleCNN` 4-Blok Ganda):**
     ```bash
     python "Train.py" --img_size 128 --epochs 50 --batch_size 32
     python Tes.py
     ```
   * **Algoritma 3 (`EfficientNetV2-S`):**
     ```bash
     python traintes.py --img_size 128 --epochs 50 --batch_size 32
     ```

---

## 👥 Anggota Kelompok 8

* **`Tessagyani Divanty Tode`** — `108102500052`
* **`Luqman Syarif Rahmadani`** — `108102530007`
* **`Rayana Bintang Rahmanda`** — `108102500034`
