# Klasifikasi Mpox Skin Lesion – Algoritma 1 (Double SimpleCNN)

## Tugas Minggu 10 Kelompok 8

| Nama | NIM | Peran |
|------|-----|-------|
| TESSAGYANI DIVANTY TODE  | 108102500052 | Algoritma 1 - Simple CNN (2 blok konvolusi) |
| LUQMAN SYARIF RAHMADANI  | 108102530007 | Algoritma 2 - Simple CNN modifikasi |
| RAYANNA BINTANG RAHMANDA | 108102500034 | Algoritma 3 - Deep learning pretrained |


Dataset: [Mpox Skin Lesion Dataset v2.0 (MSLD v2.0)](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20/data) – 6 kelas.

## Struktur
```
Models/CNN.py     # arsitektur DoubleSimpleCNN
Utils/getData.py  # loader dataset + preprocessing
train.py          # training 5-fold
test.py           # evaluasi 5-fold
Results/          # gambar training, confusion matrix, metrics.csv
```

## Cara menjalankan
```bash
pip install -r requirements.txt
# letakkan dataset di ./Dataset/Original Images dan ./Dataset/Augmented Images
python train.py
python test.py
```

## Arsitektur
(isi: gambar/tabel layer, jelaskan 2x SimpleCNN, ukuran input, loss, optimizer)

## Preprocessing
(isi: resize 128x128, konversi RGB, CLAHE, normalisasi 0-1)

## Hasil Training
![training](Results/training_fold1.png)

## Hasil Testing (rerata 5-fold)
| Metrik | Nilai | Std |
|---|---|---|
| Accuracy | 0.4905 | 0.0941 |
| Precision | 0.5449 | 0.0549 |
| Recall | 0.4905 | 0.0941 |
| F1 | 0.4880 | 0.0852 |
| ROC-AUC | 0.6982 | 0.0509 |

![confusion matrix](Results/confusion_matrix.png)

## Analisis
(isi: kelas mana yang sering salah, overfitting atau tidak, dll.)
