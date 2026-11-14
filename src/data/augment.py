"""Training-time augmentations for E3, applied batch-wise on the GPU.

Input is a standardized batch (B, leads, T), so amplitudes below are in units of
per-lead standard deviations. Each augmentation is applied independently to each
record with probability P. Validation and test data are never augmented.

    noise    additive Gaussian noise, sigma ~ U(0, 0.05)
    wander   baseline wander: per-lead sinusoid, 0.05-0.5 Hz, amplitude ~ U(0, 0.3)
    scale    amplitude scaling, one factor ~ U(0.8, 1.2) per record
    shift    circular time shift by up to +-1 s
"""

import math

import torch

P = 0.5
NOISE_SIGMA = 0.05
WANDER_FREQ_HZ = (0.05, 0.5)
WANDER_AMP = 0.3
SCALE_RANGE = (0.8, 1.2)
SHIFT_MAX_S = 1.0

AVAILABLE = ("noise", "wander", "scale", "shift")


def parse_augment(spec: str) -> list[str]:
    if spec in ("", "none", None):
        return []
    names = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [n for n in names if n not in AVAILABLE]
    if bad:
        raise ValueError(f"unknown augmentation(s) {bad}; available: {AVAILABLE}")
    return names


class Augmenter:
    def __init__(self, spec: str, fs: int, generator: torch.Generator | None = None):
        self.names = parse_augment(spec)
        self.fs = fs
        self.g = generator

    def __bool__(self):
        return bool(self.names)

    def _rand(self, *shape, device):
        return torch.rand(*shape, device=device, generator=self.g)

    def _mask(self, b, device):
        return (self._rand(b, 1, 1, device=device) < P).float()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        dev = x.device
        for name in self.names:
            m = self._mask(b, dev)
            if name == "noise":
                sigma = self._rand(b, 1, 1, device=dev) * NOISE_SIGMA
                x = x + m * sigma * torch.randn(x.shape, device=dev, generator=self.g)
            elif name == "wander":
                lo, hi = WANDER_FREQ_HZ
                f = lo + (hi - lo) * self._rand(b, c, 1, device=dev)
                phase = 2 * math.pi * self._rand(b, c, 1, device=dev)
                amp = WANDER_AMP * self._rand(b, c, 1, device=dev)
                time = torch.arange(t, device=dev).view(1, 1, t) / self.fs
                x = x + m * amp * torch.sin(2 * math.pi * f * time + phase)
            elif name == "scale":
                lo, hi = SCALE_RANGE
                s = lo + (hi - lo) * self._rand(b, 1, 1, device=dev)
                x = x * (1 + m * (s - 1))
            elif name == "shift":
                max_shift = int(SHIFT_MAX_S * self.fs)
                k = torch.randint(-max_shift, max_shift + 1, (b,), device=dev, generator=self.g)
                k = k * m.view(b).long()
                idx = (torch.arange(t, device=dev).view(1, t) - k.view(b, 1)) % t
                x = torch.gather(x, 2, idx.view(b, 1, t).expand(b, c, t))
        return x
