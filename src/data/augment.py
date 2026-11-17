"""Training-time augmentations for E3, applied batch-wise on the GPU.

Input is a standardized batch (B, leads, T), so amplitudes below are in units of
per-lead standard deviations. Validation and test data are never augmented.

    noise    additive Gaussian noise, sigma ~ U(0.05, 0.15)  (SNR about 16-26 dB)
    wander   baseline wander: per-lead sinusoid, 0.05-0.5 Hz, amplitude ~ U(0.1, 0.5)
    scale    amplitude scaling, one factor ~ U(0.8, 1.2) per record
    shift    circular time shift by up to +-1 s
    crop     random window of CROP_S = 2.5 s

noise, wander, scale and shift are each applied independently to each record with
probability P. crop is applied to *every* record (all inputs of a batch must have the
same length) and always last, so the other augmentations act on the full record
first.

A model trained with crop sees only CROP_S-second inputs, so it is also validated and
tested on windows of that length: each record is cut into sliding windows (stride
CROP_S / 2, i.e. 7 windows for a 10 s record) and the per-class maximum over the
windows is the record's prediction, as in the PTB-XL benchmark (Strodthoff et al.,
2021). See eval_window() and src.evaluation.evaluate.predict_logits.

Strengths were set by visual inspection of augmented records (clearly visible, still
physiologically plausible), not by validation or test performance. Lower bounds
above 0 make sure an applied augmentation is never a near no-op. The crop length is
the one used in the benchmark.
"""

import math

import torch

P = 0.5
NOISE_SIGMA_RANGE = (0.05, 0.15)
WANDER_FREQ_HZ = (0.05, 0.5)
WANDER_AMP_RANGE = (0.1, 0.5)
SCALE_RANGE = (0.8, 1.2)
SHIFT_MAX_S = 1.0
CROP_S = 2.5
EVAL_STRIDE_FRACTION = 0.5  # window stride at evaluation, as a fraction of CROP_S

AVAILABLE = ("noise", "wander", "scale", "shift", "crop")


def parse_augment(spec: str) -> list[str]:
    if spec in ("", "none", None):
        return []
    names = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [n for n in names if n not in AVAILABLE]
    if bad:
        raise ValueError(f"unknown augmentation(s) {bad}; available: {AVAILABLE}")
    return names


def eval_window(spec: str, fs: int) -> int | None:
    """Window length (samples) for evaluating a model trained with `spec`; None = full record."""
    return int(round(CROP_S * fs)) if "crop" in parse_augment(spec) else None


class Augmenter:
    def __init__(self, spec: str, fs: int, generator: torch.Generator | None = None):
        self.names = parse_augment(spec)
        self.fs = fs
        self.g = generator

    def __bool__(self):
        return bool(self.names)

    def _rand(self, *shape, device):
        return torch.rand(*shape, device=device, generator=self.g)

    def _uniform(self, lo_hi, *shape, device):
        lo, hi = lo_hi
        return lo + (hi - lo) * self._rand(*shape, device=device)

    def _mask(self, b, device):
        return (self._rand(b, 1, 1, device=device) < P).float()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        dev = x.device
        for name in self.names:
            if name == "crop":
                continue  # applied last, below
            m = self._mask(b, dev)
            if name == "noise":
                sigma = self._uniform(NOISE_SIGMA_RANGE, b, 1, 1, device=dev)
                x = x + m * sigma * torch.randn(x.shape, device=dev, generator=self.g)
            elif name == "wander":
                f = self._uniform(WANDER_FREQ_HZ, b, c, 1, device=dev)
                phase = 2 * math.pi * self._rand(b, c, 1, device=dev)
                amp = self._uniform(WANDER_AMP_RANGE, b, c, 1, device=dev)
                time = torch.arange(t, device=dev).view(1, 1, t) / self.fs
                x = x + m * amp * torch.sin(2 * math.pi * f * time + phase)
            elif name == "scale":
                s = self._uniform(SCALE_RANGE, b, 1, 1, device=dev)
                x = x * (1 + m * (s - 1))
            elif name == "shift":
                max_shift = int(SHIFT_MAX_S * self.fs)
                k = torch.randint(-max_shift, max_shift + 1, (b,), device=dev, generator=self.g)
                k = k * m.view(b).long()
                idx = (torch.arange(t, device=dev).view(1, t) - k.view(b, 1)) % t
                x = torch.gather(x, 2, idx.view(b, 1, t).expand(b, c, t))
        if "crop" in self.names:
            length = int(round(CROP_S * self.fs))
            start = torch.randint(0, t - length + 1, (b,), device=dev, generator=self.g)
            idx = start.view(b, 1) + torch.arange(length, device=dev).view(1, length)
            x = torch.gather(x, 2, idx.view(b, 1, length).expand(b, c, length))
        return x
