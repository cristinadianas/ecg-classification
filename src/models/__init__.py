"""Model registry. New architectures (B0 CNN, BiLSTM, CNN+Transformer for E2) are
added to MODELS; every builder takes (in_channels, num_classes, **kwargs) and returns
a module mapping (batch, leads, time) -> (batch, num_classes) logits."""

from torch import nn

from src.models.fcn1d import fcn1d
from src.models.resnet1d import resnet1d18, resnet1d34

MODELS = {
    "fcn1d": fcn1d,
    "resnet1d18": resnet1d18,
    "resnet1d34": resnet1d34,
}


def build_model(name: str, in_channels: int, num_classes: int, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name](in_channels, num_classes, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
