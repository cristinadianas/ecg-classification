"""CNN + Transformer (E2).

A convolutional front-end extracts local waveform features and shortens the sequence;
a Transformer encoder then relates all positions to each other through self-attention.
The front-end is the beginning of the S1/S2 ResNet (stride-1 stem with kernel 7, then
`n_down` residual blocks with stride 2, so 2.5 s at 100 Hz -> 63 tokens for n_down=2);
the head is the same concat-pooling head (heads.py), applied over the tokens.

    d_model   channels of the front-end = token dimension
    layers    Transformer encoder layers (pre-norm, GELU, Vaswani et al., 2017; Xiong et al., 2020)
    heads     attention heads
    ff_mult   feed-forward width = ff_mult * d_model
    attn_dropout  dropout inside the encoder layers
    n_down    stride-2 residual blocks in the front-end (sequence shortened by 2**n_down)
"""

import math

import torch
from torch import nn

from src.models.heads import concat_head
from src.models.resnet1d_plus import Block


class SinusoidalPositions(nn.Module):
    """Fixed sinusoidal position encoding added to the tokens (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):  # (batch, tokens, d_model)
        return x + self.pe[: x.shape[1]].to(x.dtype)


class CNNTransformer(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, d_model: int = 128, layers: int = 2,
                 heads: int = 4, ff_mult: int = 4, attn_dropout: float = 0.1, n_down: int = 2,
                 kernel_size: int = 5, stem_kernel_size: int = 7, ps_head: float = 0.5,
                 lin_ftrs_head: int = 128, dropout: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, d_model, stem_kernel_size, 1, stem_kernel_size // 2, bias=False),
            nn.BatchNorm1d(d_model), nn.ReLU(inplace=True))
        self.cnn = nn.Sequential(*[Block(d_model, d_model, 2, kernel_size, "basic", False, False, 0.0)
                                   for _ in range(n_down)])
        self.pos = SinusoidalPositions(d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff_mult * d_model, attn_dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d_model),
                                             enable_nested_tensor=False)
        self.head = concat_head(d_model, num_classes, lin_ftrs_head, ps_head)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, Block):
                nn.init.zeros_(m.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.cnn(self.stem(x)).transpose(1, 2)     # (batch, tokens, d_model)
        tokens = self.encoder(self.pos(tokens))
        return self.head(tokens.transpose(1, 2))


def cnn_transformer(in_channels: int, num_classes: int, **kwargs) -> CNNTransformer:
    return CNNTransformer(in_channels, num_classes, **kwargs)
