import torch
import torch.nn as nn


def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(5, 5), padding=2),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2)),
    )


class SimpleCNN(nn.Module):
    """SimpleCNN asli (2 conv block) -- dipakai sebagai satu 'stage'."""
    def __init__(self, input_c=3, channels=(20, 50)):
        super().__init__()
        self.block1 = conv_block(input_c, channels[0])
        self.block2 = conv_block(channels[0], channels[1])

    def forward(self, x):
        return self.block2(self.block1(x))


class DoubleSimpleCNN(nn.Module):
    """Algoritma 1: arsitektur 2x SimpleCNN (stage 1 -> stage 2) + classifier."""
    def __init__(self, input_dim=128, input_c=3, output=6, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.stage1 = SimpleCNN(input_c, channels=(20, 50))
        self.stage2 = SimpleCNN(50, channels=(100, 100))

        # hitung ukuran flatten otomatis dengan dummy input
        with torch.no_grad():
            n_flat = self.stage2(self.stage1(torch.zeros(1, input_c, input_dim, input_dim))).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(n_flat, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output),
        )

    def forward(self, x):
        x = self.stage2(self.stage1(x))
        # output = LOGITS (tanpa Softmax), karena nn.CrossEntropyLoss sudah
        # menghitung log-softmax sendiri. Softmax baru dipakai di test.py.
        return self.classifier(x)
