"""
Models/CNN.py
-------------
Simple CNN untuk klasifikasi 6 kelas citra lesi kulit (Mpox dkk).

Arsitektur:
  4 blok konvolusi [Conv-BN-ReLU x2 -> MaxPool -> Dropout2d]
  (32 -> 64 -> 128 -> 256 channel)
  -> Global Average Pooling -> FC(256->128) -> Dropout -> FC(128->num_classes)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(drop),
        )

    def forward(self, x):
        return self.block(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 6, in_channels: int = 3,
                 widths=(32, 64, 128, 256), dropout: float = 0.4):
        super().__init__()
        layers, prev = [], in_channels
        for i, w in enumerate(widths):
            layers.append(ConvBlock(prev, w, drop=0.05 + 0.05 * i))
            prev = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(prev, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = SimpleCNN(num_classes=6)
    out = model(torch.randn(2, 3, 128, 128))
    n_params = sum(p.numel() for p in model.parameters())
    print("Output:", tuple(out.shape), "| Parameter:", f"{n_params:,}")
