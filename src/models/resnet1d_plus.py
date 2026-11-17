"""Configurable 1D ResNet for the architecture search of S2.

Defaults reproduce the best custom ResNet of S1 (resnet1d10_stem1_chead_w128): stride-1
stem without max-pool, four stages of one basic block, constant width 128, kernel
size 5, concat-pooling head. Every option below stays within the ResNet family
(residual blocks with identity / projection shortcuts):

    layers        blocks per stage, e.g. (1,1,1,1) or (2,2,2,2)
    widths        output channels per stage, e.g. (128,)*4 or (64,128,256,512)
    strides       stride of the first block of each stage (default 1,2,2,2,...)
    dilations     dilation of the block convolutions per stage (dilated ResNet, Yu et al., 2017)
    kernel_size   kernel size of the block convolutions; int or (k1, k2) for the two convs
    stem          "conv": one conv (k=stem_kernel_size, stride stem_stride, optional max-pool)
                  "deep": three convs (k=5) as in ResNet-C / xresnet (He et al., 2019)
                  "leadwise": a grouped conv per lead (stem_per_lead filters each), then a
                  1x1 conv that mixes the leads
    block         "basic": conv-BN-act-conv-BN (+ shortcut) -> act (He et al., 2016a)
                  "preact": BN-act-conv-BN-act-conv (+ shortcut) (He et al., 2016b)
                  "bottleneck": 1x1 reduce, k conv, 1x1 expand (ResNet-50, He et al., 2016a)
    bottleneck_ratio  bottleneck width / block width (0.25 as in ResNet-50)
    se            squeeze-and-excitation on the residual branch (Hu et al., 2018)
    resnet_d      average-pool before the 1x1 projection shortcut (ResNet-D, He et al., 2019)
    downsample    "conv": strided convolution; "blur": stride-1 convolution followed by a
                  low-pass blur-pool, also in the shortcut (anti-aliasing, Zhang, 2019)
    act           "relu" or "silu"
    drop_path     stochastic depth: max drop rate of residual branches, linear in depth
                  (Huang et al., 2016); 0 = off
    head          "concat" (max + avg pooling, BN, dropout; heads.py) or "avg"
"""

import torch
import torch.nn.functional as F
from torch import nn

from src.models.heads import avg_head, concat_head

ACTS = {"relu": lambda: nn.ReLU(inplace=True), "silu": lambda: nn.SiLU(inplace=True)}


class SqueezeExcite1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                nn.Linear(channels, hidden), nn.ReLU(inplace=True),
                                nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1)


class BlurPool1d(nn.Module):
    """Low-pass filter [1, 2, 1] / 4 followed by subsampling (Zhang, 2019)."""

    def __init__(self, channels: int, stride: int):
        super().__init__()
        self.channels, self.stride = channels, stride
        self.register_buffer("filt", torch.tensor([1.0, 2.0, 1.0]).div(4).repeat(channels, 1, 1),
                             persistent=False)

    def forward(self, x):
        x = F.pad(x, (1, 1), mode="reflect")
        return F.conv1d(x, self.filt.to(x.dtype), stride=self.stride, groups=self.channels)


def drop_path(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Drop the whole residual branch per sample with probability p (stochastic depth)."""
    if p == 0.0 or not training:
        return x
    keep = torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype) >= p
    return x * keep / (1 - p)


def _conv(ci, co, k, stride=1, dilation=1, groups=1):
    return nn.Conv1d(ci, co, k, stride, dilation * (k - 1) // 2, dilation=dilation, groups=groups, bias=False)


class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, kernel_size, block: str,
                 se: bool, resnet_d: bool, drop_path_p: float, dilation: int = 1,
                 downsample: str = "conv", act: str = "relu", bottleneck_ratio: float = 0.25):
        super().__init__()
        k1, k2 = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.preact = block == "preact"
        self.bottleneck = block == "bottleneck"
        blur = downsample == "blur" and stride != 1
        conv_stride = 1 if blur else stride
        self.act = ACTS[act]()
        if self.preact:
            self.bn0 = nn.BatchNorm1d(in_ch)
        if self.bottleneck:
            mid = max(int(out_ch * bottleneck_ratio), 8)
            self.conv1 = _conv(in_ch, mid, 1)
            self.bn1 = nn.BatchNorm1d(mid)
            self.conv2 = _conv(mid, mid, k1, conv_stride, dilation)
            self.bn2 = nn.BatchNorm1d(mid)
            self.conv3 = _conv(mid, out_ch, 1)
            self.bn3 = nn.BatchNorm1d(out_ch)
        else:
            self.conv1 = _conv(in_ch, out_ch, k1, conv_stride, dilation)
            self.bn1 = nn.BatchNorm1d(out_ch)
            self.conv2 = _conv(out_ch, out_ch, k2, 1, dilation)
            self.bn2 = None if self.preact else nn.BatchNorm1d(out_ch)
        self.blur = BlurPool1d(mid if self.bottleneck else out_ch, stride) if blur else None
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            proj = [nn.Conv1d(in_ch, out_ch, 1, 1 if (resnet_d or blur) else stride, bias=False)]
            if not self.preact:
                proj.append(nn.BatchNorm1d(out_ch))
            if blur:
                proj.insert(0, BlurPool1d(in_ch, stride))
            elif resnet_d and stride != 1:
                proj.insert(0, nn.AvgPool1d(stride, stride, ceil_mode=True, count_include_pad=False))
            self.downsample = nn.Sequential(*proj)
        self.se = SqueezeExcite1d(out_ch) if se else None
        self.drop_path_p = drop_path_p

    def _down(self, x):
        return x if self.blur is None else self.blur(x)

    def forward(self, x):
        if self.preact:
            out = self.act(self.bn0(x))
            identity = x if self.downsample is None else self.downsample(out)
            out = self.conv2(self._down(self.act(self.bn1(self.conv1(out)))))
        elif self.bottleneck:
            identity = x if self.downsample is None else self.downsample(x)
            out = self.act(self.bn1(self.conv1(x)))
            out = self._down(self.act(self.bn2(self.conv2(out))))
            out = self.bn3(self.conv3(out))
        else:
            identity = x if self.downsample is None else self.downsample(x)
            out = self.bn2(self.conv2(self._down(self.act(self.bn1(self.conv1(x))))))
        if self.se is not None:
            out = self.se(out)
        out = drop_path(out, self.drop_path_p, self.training) + identity
        return out if self.preact else self.act(out)


class ResNet1dPlus(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, layers=(1, 1, 1, 1),
                 widths=(128, 128, 128, 128), strides=None, dilations=None, kernel_size=5,
                 stem: str = "conv", stem_kernel_size: int = 7, stem_stride: int = 1,
                 stem_pool: bool = False, stem_per_lead: int = 8,
                 block: str = "basic", bottleneck_ratio: float = 0.25,
                 se: bool = False, resnet_d: bool = False, downsample: str = "conv",
                 act: str = "relu", drop_path: float = 0.0, head: str = "concat",
                 ps_head: float = 0.5, lin_ftrs_head: int = 128, dropout: float = 0.0):
        super().__init__()
        if act not in ACTS:
            raise ValueError(f"act must be one of {list(ACTS)}, got {act!r}")
        if downsample not in ("conv", "blur"):
            raise ValueError(f"downsample must be 'conv' or 'blur', got {downsample!r}")
        if block not in ("basic", "preact", "bottleneck"):
            raise ValueError(f"block must be 'basic', 'preact' or 'bottleneck', got {block!r}")
        strides = strides or [1] + [2] * (len(layers) - 1)
        dilations = dilations or [1] * len(layers)
        w0 = widths[0]
        if stem == "conv":
            s = [nn.Conv1d(in_channels, w0, stem_kernel_size, stem_stride, stem_kernel_size // 2, bias=False),
                 nn.BatchNorm1d(w0), ACTS[act]()]
        elif stem == "deep":
            mid = max(w0 // 2, 32)
            s = []
            for i, (ci, co) in enumerate([(in_channels, mid), (mid, mid), (mid, w0)]):
                s += [nn.Conv1d(ci, co, 5, stem_stride if i == 0 else 1, 2, bias=False),
                      nn.BatchNorm1d(co), ACTS[act]()]
        elif stem == "leadwise":
            per_lead = in_channels * stem_per_lead
            s = [nn.Conv1d(in_channels, per_lead, stem_kernel_size, stem_stride, stem_kernel_size // 2,
                           groups=in_channels, bias=False), nn.BatchNorm1d(per_lead), ACTS[act](),
                 nn.Conv1d(per_lead, w0, 1, bias=False), nn.BatchNorm1d(w0), ACTS[act]()]
        else:
            raise ValueError(f"stem must be 'conv', 'deep' or 'leadwise', got {stem!r}")
        if stem_pool:
            s.append(nn.MaxPool1d(3, 2, 1))
        self.stem = nn.Sequential(*s)

        n_blocks = sum(layers)
        blocks, in_ch, k = [], w0, 0
        for n, width, stride, dil in zip(layers, widths, strides, dilations):
            for b in range(n):
                p = drop_path * k / max(n_blocks - 1, 1)
                blocks.append(Block(in_ch, width, stride if b == 0 else 1, kernel_size, block, se, resnet_d,
                                    p, dil, downsample, act, bottleneck_ratio))
                in_ch, k = width, k + 1
        self.stages = nn.Sequential(*blocks)
        self.post = nn.Sequential(nn.BatchNorm1d(in_ch), ACTS[act]()) if block == "preact" else nn.Identity()
        if head == "concat":
            self.head = concat_head(in_ch, num_classes, lin_ftrs_head, ps_head)
        elif head == "avg":
            self.head = avg_head(in_ch, num_classes, dropout)
        else:
            raise ValueError(f"head must be 'concat' or 'avg', got {head!r}")
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.modules():  # every residual branch starts as (close to) zero
            if isinstance(m, Block):
                last_bn = m.bn3 if m.bottleneck else m.bn2
                if last_bn is not None:
                    nn.init.zeros_(last_bn.weight)
                else:
                    nn.init.zeros_(m.conv2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.post(self.stages(self.stem(x))))


def resnet1d_plus(in_channels: int, num_classes: int, **kwargs) -> ResNet1dPlus:
    return ResNet1dPlus(in_channels, num_classes, **kwargs)
