"""Classification heads shared by the ResNet variants."""

import torch
from torch import nn


class AdaptiveConcatPool1d(nn.Module):
    """Global max pooling and global average pooling, concatenated (fastai)."""

    def __init__(self):
        super().__init__()
        self.max = nn.AdaptiveMaxPool1d(1)
        self.avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        return torch.cat([self.max(x), self.avg(x)], dim=1)


def _mlp(n_in: int, num_classes: int, lin_ftrs: int, ps: float) -> list[nn.Module]:
    return [nn.BatchNorm1d(n_in), nn.Dropout(ps / 2), nn.Linear(n_in, lin_ftrs), nn.ReLU(inplace=True),
            nn.BatchNorm1d(lin_ftrs), nn.Dropout(ps), nn.Linear(lin_ftrs, num_classes)]


def concat_head(n_features: int, num_classes: int, lin_ftrs: int = 128, ps: float = 0.5) -> nn.Sequential:
    """Head of the PTB-XL benchmark models (fastai create_head1d with lin_ftrs=[128], ps=0.5):
    concat(max pool, avg pool) -> BN -> Dropout(ps/2) -> Linear(2n, lin_ftrs) -> ReLU
    -> BN -> Dropout(ps) -> Linear(lin_ftrs, classes)."""
    return nn.Sequential(AdaptiveConcatPool1d(), nn.Flatten(), *_mlp(2 * n_features, num_classes, lin_ftrs, ps))


def mlp_head(n_in: int, num_classes: int, lin_ftrs: int = 128, ps: float = 0.5) -> nn.Sequential:
    """The part of concat_head after pooling, for models that pool themselves (BiLSTM)."""
    return nn.Sequential(*_mlp(n_in, num_classes, lin_ftrs, ps))


def avg_head(n_features: int, num_classes: int, dropout: float = 0.0) -> nn.Sequential:
    """Classic ResNet head: global average pooling -> (dropout) -> Linear."""
    return nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout),
                         nn.Linear(n_features, num_classes))
