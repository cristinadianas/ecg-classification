"""1D ResNet (He et al., 2016) for multi-lead ECG.

The 2D ResNet-18/34 layout carried over to 1D: a strided stem, four stages of basic
blocks with widths 64-128-256-512, each later stage halving the time axis, global
average pooling and a linear head with one logit per class.

ResNet1D-18 and ResNet1D-34 differ *only* in the number of blocks per stage
([2,2,2,2] vs [3,4,6,3]); kernel size, widths, stem and head are identical, so E1
isolates depth.

Shapes at 100 Hz, 12 leads: input (B, 12, 1000) -> stem (B, 64, 250) ->
stages 250 / 125 / 63 / 32 time steps -> logits (B, 5). At 500 Hz every
length is 5x longer; nothing in the model depends on the input length.
"""

import torch
from torch import nn


class BasicBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, 1, pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm1d(out_ch))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1d(nn.Module):
    def __init__(self, layers, in_channels: int, num_classes: int,
                 widths=(64, 128, 256, 512), kernel_size: int = 5, stem_kernel_size: int = 7,
                 dropout: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, widths[0], stem_kernel_size, 2, stem_kernel_size // 2, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2, 1),
        )
        stages, in_ch = [], widths[0]
        for i, (n_blocks, width) in enumerate(zip(layers, widths)):
            blocks = []
            for b in range(n_blocks):
                stride = 2 if (i > 0 and b == 0) else 1
                blocks.append(BasicBlock1d(in_ch, width, stride, kernel_size))
                in_ch = width
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, num_classes))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero-init the last BN of each residual branch so every block starts as the
        # identity (Goyal et al., 2017). Helps the deeper net train as easily as the
        # shallow one, which matters for a fair depth comparison.
        for m in self.modules():
            if isinstance(m, BasicBlock1d):
                nn.init.zeros_(m.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(self.stem(x)))


def resnet1d18(in_channels: int, num_classes: int, **kwargs) -> ResNet1d:
    return ResNet1d([2, 2, 2, 2], in_channels, num_classes, **kwargs)


def resnet1d34(in_channels: int, num_classes: int, **kwargs) -> ResNet1d:
    return ResNet1d([3, 4, 6, 3], in_channels, num_classes, **kwargs)
