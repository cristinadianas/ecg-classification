"""FCN of Wang et al. (2017) for multi-lead ECG (B0 baseline)."""

import torch
from torch import nn


class ConvBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        self.pad = nn.ConstantPad1d(((kernel_size - 1) // 2, kernel_size // 2), 0.0)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(self.pad(x))))


class FCN1d(nn.Module):
    def __init__(self, in_channels: int, num_classes: int,
                 widths=(128, 256, 128), kernel_sizes=(8, 5, 3), dropout: float = 0.0):
        super().__init__()
        blocks, in_ch = [], in_channels
        for width, k in zip(widths, kernel_sizes):
            blocks.append(ConvBlock1d(in_ch, width, k))
            in_ch = width
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def fcn1d(in_channels: int, num_classes: int, **kwargs) -> FCN1d:
    return FCN1d(in_channels, num_classes, **kwargs)
