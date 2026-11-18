import os
import re
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent          # folder project (isi train.py, test.py)
DATASET_DIR = ROOT / "Dataset"
ORIG_DIR = DATASET_DIR / "Original Images"
AUG_DIR = DATASET_DIR / "Augmented Images"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUM_FOLDS = 5

# Normalisasi piksel ke rentang [-1, 1]
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


# Helper path
@lru_cache(maxsize=None)
def find_split(root: str, fold: int, split: str) -> Path:
    """Cari folder <split> (Train/Valid/Test) milik fold ke-<fold> di dalam <root>.
    Tahan terhadap perbedaan nama folder (Fold1, fold1, fold1_AUG, folder bersarang)."""
    pattern = re.compile(rf"fold_?{fold}(?!\d)", re.IGNORECASE)
    for path, _, _ in os.walk(root):
        rel = os.path.relpath(path, root).replace("\\", "/")
        if os.path.basename(path).lower() == split.lower() and pattern.search(rel):
            return Path(path)
    raise FileNotFoundError(f"Folder '{split}' untuk fold {fold} tidak ditemukan di {root}")


def orig_split_dir(fold: int, split: str) -> Path:
    """split: 'Train' | 'Valid' | 'Test'"""
    return find_split(str(ORIG_DIR), fold, split)


def aug_train_dir(fold: int) -> Path:
    return find_split(str(AUG_DIR), fold, "Train")


def get_class_names(fold: int = 1):
    """Nama kelas diambil dari nama subfolder (diurutkan alfabetis)."""
    d = orig_split_dir(fold, "Train")
    return sorted(p.name for p in d.iterdir() if p.is_dir())


# Dataset
class MpoxDataset(Dataset):
    def __init__(self, roots, class_names, transform=None):
        self.class_names = list(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.transform = transform
        self.samples = []

        for root in roots:
            root = Path(root)
            if not root.is_dir():
                print(f"[PERINGATAN] Folder tidak ada, dilewati: {root}")
                continue
            for cls, idx in self.class_to_idx.items():
                cdir = root / cls
                if not cdir.is_dir():
                    print(f"[PERINGATAN] Kelas '{cls}' tidak ada di {root}")
                    continue
                for p in sorted(cdir.iterdir()):
                    if p.suffix.lower() in IMG_EXT:
                        self.samples.append((str(p), idx))

        if not self.samples:
            raise RuntimeError(f"Tidak ada gambar yang ditemukan di: {[str(r) for r in roots]}")
        self.targets = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# Pre-processing
def get_transforms(img_size: int = 128):
    """
    Train : resize -> flip -> rotasi kecil -> color jitter ringan -> tensor -> normalisasi
            (augmentasi tambahan 'on-the-fly' di atas augmented images yang sudah ada)
    Valid/Test : resize -> tensor -> normalisasi (tanpa augmentasi)
    """
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def compute_class_weights(targets, num_classes):
    """Bobot kelas berbanding terbalik dengan frekuensi (untuk CrossEntropyLoss)."""
    counts = torch.bincount(torch.tensor(targets), minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    return weights


def get_dataloaders(fold: int, img_size: int = 128, batch_size: int = 32,
                    num_workers: int = 4, use_augmented: bool = True,
                    use_sampler: bool = False):
    class_names = get_class_names(fold)
    train_tf, eval_tf = get_transforms(img_size)

    train_roots = [orig_split_dir(fold, "Train")]
    if use_augmented:
        train_roots.append(aug_train_dir(fold))

    train_ds = MpoxDataset(train_roots, class_names, train_tf)
    valid_ds = MpoxDataset([orig_split_dir(fold, "Valid")], class_names, eval_tf)
    test_ds = MpoxDataset([orig_split_dir(fold, "Test")], class_names, eval_tf)

    pin = torch.cuda.is_available()
    persistent = num_workers > 0

    class_weights = compute_class_weights(train_ds.targets, len(class_names))

    sampler = None
    if use_sampler:
        sample_w = [class_weights[t].item() for t in train_ds.targets]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=(sampler is None), sampler=sampler,
                              num_workers=num_workers, pin_memory=pin,
                              persistent_workers=persistent, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin,
                              persistent_workers=persistent)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin,
                             persistent_workers=persistent)

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "class_names": class_names,
        "class_weights": class_weights,
        "sizes": (len(train_ds), len(valid_ds), len(test_ds)),
    }


def get_test_loader(fold: int, img_size: int = 128, batch_size: int = 32, num_workers: int = 4):
    """Hanya test loader (dipakai test.py)."""
    class_names = get_class_names(fold)
    _, eval_tf = get_transforms(img_size)
    test_ds = MpoxDataset([orig_split_dir(fold, "Test")], class_names, eval_tf)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return loader, class_names


if __name__ == "__main__":
    print(f"ROOT : {ROOT}")
    names = get_class_names(1)
    print(f"Kelas: {names}\n")
    for k in range(1, NUM_FOLDS + 1):
        tr_o = MpoxDataset([orig_split_dir(k, "Train")], names)
        tr_a = MpoxDataset([aug_train_dir(k)], names)
        va = MpoxDataset([orig_split_dir(k, "Valid")], names)
        te = MpoxDataset([orig_split_dir(k, "Test")], names)
        print(f"Fold {k}: train_orig={len(tr_o)} train_aug={len(tr_a)} valid={len(va)} test={len(te)}")
