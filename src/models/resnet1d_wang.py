"""Reproduction of `resnet1d_wang` from the PTB-XL benchmark (Strodthoff et al., 2021).

The 1D ResNet of Wang et al. (2017) as implemented in the benchmark code
(github.com/helme/ecg_ptbxl_benchmarking, models/resnet1d.py and basic_conv1d.py):

    stem    Conv1d(k=7, stride 1, no pooling) -> BN -> ReLU                    128 channels
    stages  3 x one basic block, convs k=5 then k=3, strides 1 / 2 / 2,          128 channels
            1x1 conv + BN shortcut where the stride changes
    head    concat(adaptive max pool, adaptive avg pool) -> BN -> Dropout(0.25)
            -> Linear(256, 128) -> ReLU -> BN -> Dropout(0.5) -> Linear(128, classes)

Weights use the PyTorch default initialization, as in the benchmark (no custom init,
no zero-initialized residual branches). About 0.5M parameters for 12 leads.

Shapes at 100 Hz, 12 leads, 2.5 s crops: (B, 12, 250) -> stem (B, 128, 250) ->
stages 250 / 125 / 63 time steps -> logits (B, 5).
"""

import torch
from torch import nn

from src.models.heads import concat_head
from src.models.resnet1d import BasicBlock1d


class ResNet1dWang(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, width: int = 128,
                 layers=(1, 1, 1), kernel_size=(5, 3), stem_kernel_size: int = 7,
                 lin_ftrs: int = 128, ps_head: float = 0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, stem_kernel_size, 1, (stem_kernel_size - 1) // 2, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for i, n_blocks in enumerate(layers):
            for b in range(n_blocks):
                stride = 2 if (i > 0 and b == 0) else 1
                blocks.append(BasicBlock1d(width, width, stride, kernel_size))
        self.stages = nn.Sequential(*blocks)
        # fastai create_head1d with lin_ftrs=[128], ps=0.5: hidden layer gets ps/2
        self.head = concat_head(width, num_classes, lin_ftrs, ps_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(self.stem(x)))


def resnet1d_wang(in_channels: int, num_classes: int, dropout: float | None = None,
                  **kwargs) -> ResNet1dWang:
    """`dropout` (the generic --dropout flag) does not apply: the head dropout is fixed
    by `ps_head` (0.5) as in the benchmark. Pass ps_head via --model-kwargs to change it."""
    return ResNet1dWang(in_channels, num_classes, **kwargs)
