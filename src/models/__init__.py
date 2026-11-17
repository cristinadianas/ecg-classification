"""Model registry. New architectures are added to MODELS; every builder takes
(in_channels, num_classes, **kwargs) and returns a module mapping
(batch, leads, time) -> (batch, num_classes) logits."""

from torch import nn

from src.models.bilstm import bilstm
from src.models.cnn_transformer import cnn_transformer
from src.models.resnet1d import (
    resnet1d10, resnet1d10_chead, resnet1d10_stem1, resnet1d10_stem1_chead_w128, resnet1d10_w128,
    resnet1d18, resnet1d34,
)
from src.models.resnet1d_plus import resnet1d_plus
from src.models.resnet1d_wang import resnet1d_wang

MODELS = {
    # classic ResNet layout (E1)
    "resnet1d10": resnet1d10,
    "resnet1d18": resnet1d18,
    "resnet1d34": resnet1d34,
    # custom ResNets: ResNet1D-10 with one design choice changed (S1)
    "resnet1d10_stem1": resnet1d10_stem1,                          # stride-1 stem, no max-pool
    "resnet1d10_chead": resnet1d10_chead,                          # max + avg pooling, dropout head
    "resnet1d10_w128": resnet1d10_w128,                            # constant width 128
    "resnet1d10_stem1_chead_w128": resnet1d10_stem1_chead_w128,    # all three
    # configurable ResNet for the architecture search (S2); variants via --model-kwargs
    "resnet1d_plus": resnet1d_plus,
    # benchmark model (B0, S1)
    "resnet1d_wang": resnet1d_wang,
    # other operation types (E2); sizes via --model-kwargs
    "bilstm": bilstm,
    "cnn_transformer": cnn_transformer,
}


def build_model(name: str, in_channels: int, num_classes: int, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name](in_channels, num_classes, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
