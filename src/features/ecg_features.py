"""Hand-crafted per-lead ECG features for the SVM baseline (B0).

The two extractors are taken from the original notebooks in src/models/, which are
kept for reference:

    statistical  39 features per lead (SVM.ipynb): amplitude statistics, percentiles,
                 distribution shape, first/second differences, Hjorth parameters and
                 Welch power-spectrum descriptors. No beat detection.
    wavelet      62 features per lead (SVM_ECG_Wavelet_Lead_Comparison.ipynb): the same
                 kind of statistics after band-pass filtering, approximate R-peaks and
                 RR intervals, a median R-aligned beat summarized before / around / after
                 the QRS complex, and a 4-level Daubechies-4 wavelet decomposition.

The feature code is unchanged except for two points that do not change the values:
np.trapz (deprecated) is replaced by np.trapezoid, and the wavelet decomposition
(pywt.wavedec, mode "symmetric") is reimplemented in NumPy below, because the compiled
PyWavelets extension cannot be loaded on every machine of the project.

Features are computed from the raw millivolt signals of the processed cache
(src.data.dataset.load_splits), independently for every lead, so a lead subset is
simply a subset of the per-lead feature blocks. They are computed once per sampling
rate and feature set and cached in data/processed/features_<kind>_<fs>hz.npz.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.signal import butter, find_peaks, peak_widths, sosfiltfilt, welch
from scipy.stats import kurtosis, skew

from src.config import PROCESSED_DIR

# ----------------------------------------------------------------------------- wavelet transform

# Daubechies-4 decomposition filters, as tabulated in PyWavelets (pywt.Wavelet("db4")).
DB4_DEC_LO = np.array([-0.010597401784997278, 0.032883011666982945, 0.030841381835986965,
                       -0.18703481171888114, -0.02798376941698385, 0.6308807679295904,
                       0.7148465705525415, 0.23037781330885523])
DB4_DEC_HI = np.array([-0.23037781330885523, 0.7148465705525415, -0.6308807679295904,
                       -0.02798376941698385, 0.18703481171888114, 0.030841381835986965,
                       -0.032883011666982945, -0.010597401784997278])


def dwt(x: np.ndarray, dec_lo: np.ndarray, dec_hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One level of pywt.dwt(x, wavelet, mode="symmetric").

    PyWavelets extends the signal by len(filter) - 1 samples on each side with
    half-sample symmetric extension, convolves with the decomposition filters and
    keeps every second sample starting at index 1; the output length is
    floor((len(x) + len(filter) - 1) / 2).
    """
    padded = np.pad(x, len(dec_lo) - 1, mode="symmetric")
    return (np.convolve(padded, dec_lo, mode="valid")[1::2],
            np.convolve(padded, dec_hi, mode="valid")[1::2])


def dwt_max_level(n: int, filter_len: int) -> int:
    """pywt.dwt_max_level."""
    if filter_len <= 1 or n < filter_len - 1:
        return 0
    return int(np.floor(np.log2(n // (filter_len - 1))))


def wavedec(x: np.ndarray, level: int, dec_lo=DB4_DEC_LO, dec_hi=DB4_DEC_HI) -> list[np.ndarray]:
    """pywt.wavedec(x, "db4", mode="symmetric", level=level) -> [cA_n, cD_n, ..., cD_1]."""
    details, approx = [], np.asarray(x, dtype=np.float64)
    for _ in range(level):
        approx, detail = dwt(approx, dec_lo, dec_hi)
        details.append(detail)
    return [approx] + details[::-1]


# ----------------------------------------------------------------------------- statistical (SVM.ipynb)

STATISTICAL_NAMES = [
    "mean", "std", "min", "max", "peak_to_peak", "rms",
    "p10", "p25", "p50", "p75", "p90", "skewness", "kurtosis",
    "mean_abs_first_diff", "std_first_diff", "max_abs_first_diff",
    "median_abs_deviation", "interquartile_range", "p05", "p95",
    "mean_abs_amplitude", "waveform_length_per_interval",
    "zero_crossing_rate", "slope_sign_change_rate",
    "first_diff_rms", "second_diff_rms", "crest_factor", "shape_factor",
    "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
    "total_power_0p5_40Hz", "relative_power_0p5_5Hz",
    "relative_power_5_15Hz", "relative_power_15_40Hz",
    "spectral_centroid", "dominant_frequency", "spectral_entropy",
    "spectral_rolloff_95",
]


def statistical_features(signal, fs=100):
    """39 descriptive features of one 10 s lead (SVM.ipynb, extract_lead_features)."""
    x = np.asarray(signal, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    dx = np.diff(x)
    ddx = np.diff(dx)

    # amplitude and distribution (1-13)
    mean = np.mean(x)
    std = np.std(x)
    minimum = np.min(x)
    maximum = np.max(x)
    peak_to_peak = np.ptp(x)
    rms = np.sqrt(np.mean(x ** 2))
    p10, p25, p50, p75, p90 = np.percentile(x, [10, 25, 50, 75, 90])
    signal_skew = skew(x, bias=False)
    signal_kurtosis = kurtosis(x, bias=False)

    # first-difference statistics (14-16)
    mean_abs_diff = np.mean(np.abs(dx)) if dx.size else 0.0
    std_diff = np.std(dx) if dx.size else 0.0
    max_abs_diff = np.max(np.abs(dx)) if dx.size else 0.0

    # robust amplitude and spread (17-21)
    median_abs_deviation = np.median(np.abs(x - p50))
    interquartile_range = p75 - p25
    p05, p95 = np.percentile(x, [5, 95])
    mean_abs_amplitude = np.mean(np.abs(x))

    # waveform shape and variation (22-28)
    waveform_length = np.sum(np.abs(dx)) / max(len(x) - 1, 1)
    centered = x - mean
    zero_crossing_rate = np.mean(centered[:-1] * centered[1:] < 0) if len(x) > 1 else 0.0
    slope_sign_change_rate = np.mean(dx[:-1] * dx[1:] < 0) if len(dx) > 1 else 0.0
    first_diff_rms = np.sqrt(np.mean(dx ** 2)) if dx.size else 0.0
    second_diff_rms = np.sqrt(np.mean(ddx ** 2)) if ddx.size else 0.0
    crest_factor = np.max(np.abs(x)) / (rms + 1e-12)
    shape_factor = rms / (mean_abs_amplitude + 1e-12)

    # Hjorth descriptors (29-31)
    activity = np.var(x)
    mobility = np.sqrt(np.var(dx) / (activity + 1e-12)) if dx.size else 0.0
    derivative_mobility = np.sqrt(np.var(ddx) / (np.var(dx) + 1e-12)) if ddx.size else 0.0
    hjorth_complexity = derivative_mobility / (mobility + 1e-12)

    # Welch power spectral density (32-39)
    frequencies, psd = welch(x, fs=fs, nperseg=min(256, len(x)), detrend="constant", scaling="density")
    band_mask = (frequencies >= 0.5) & (frequencies <= 40.0)
    band_freqs = frequencies[band_mask]
    band_psd = psd[band_mask]
    total_power = np.trapezoid(band_psd, band_freqs) if band_psd.size > 1 else 0.0

    def relative_band_power(low, high):
        mask = (frequencies >= low) & (frequencies < high)
        power = np.trapezoid(psd[mask], frequencies[mask]) if np.count_nonzero(mask) > 1 else 0.0
        return power / (total_power + 1e-12)

    rel_power_0p5_5 = relative_band_power(0.5, 5.0)
    rel_power_5_15 = relative_band_power(5.0, 15.0)
    rel_power_15_40 = relative_band_power(15.0, 40.0)
    if band_psd.size and np.sum(band_psd) > 0:
        spectral_centroid = np.sum(band_freqs * band_psd) / np.sum(band_psd)
        dominant_frequency = band_freqs[np.argmax(band_psd)]
        normalized_psd = band_psd / np.sum(band_psd)
        spectral_entropy = -np.sum(normalized_psd * np.log(normalized_psd + 1e-12)) / np.log(len(normalized_psd) + 1e-12)
        cumulative_power = np.cumsum(band_psd)
        spectral_rolloff_95 = band_freqs[
            min(np.searchsorted(cumulative_power, 0.95 * cumulative_power[-1]), len(band_freqs) - 1)]
    else:
        spectral_centroid = dominant_frequency = spectral_entropy = spectral_rolloff_95 = 0.0

    return np.asarray([
        mean, std, minimum, maximum, peak_to_peak, rms,
        p10, p25, p50, p75, p90, signal_skew, signal_kurtosis,
        mean_abs_diff, std_diff, max_abs_diff,
        median_abs_deviation, interquartile_range, p05, p95,
        mean_abs_amplitude, waveform_length, zero_crossing_rate,
        slope_sign_change_rate, first_diff_rms, second_diff_rms,
        crest_factor, shape_factor, activity, mobility,
        hjorth_complexity, total_power, rel_power_0p5_5,
        rel_power_5_15, rel_power_15_40, spectral_centroid,
        dominant_frequency, spectral_entropy, spectral_rolloff_95,
    ], dtype=np.float32)


# ----------------------------------------------------------------------------- ECG + wavelet (lead-comparison notebook)

def _safe_float(value, default=0.0):
    value = float(value)
    return value if np.isfinite(value) else float(default)


def _coefficient_entropy(coeff):
    energy = np.square(np.asarray(coeff, dtype=np.float64))
    total = energy.sum()
    if total <= 0 or not np.isfinite(total):
        return 0.0
    probabilities = energy / total
    probabilities = probabilities[probabilities > 0]
    return _safe_float(-np.sum(probabilities * np.log(probabilities + 1e-12)))


WAVELET_NAMES = [
    "mean", "std", "rms", "median", "mad", "min", "max", "peak_to_peak",
    "q10", "q25", "q75", "q90", "skew", "kurtosis",
    "mean_abs_diff", "std_diff", "line_length", "zero_crossing_rate",
    "rr_mean_s", "rr_std_s", "rr_median_s", "rr_iqr_s", "rr_min_s", "rr_max_s",
    "rmssd_s", "mean_heart_rate_bpm",
    "peak_abs_median", "peak_abs_iqr", "peak_prominence_median", "peak_width_median_s",
    "pre_qrs_mean", "pre_qrs_std", "pre_qrs_min", "pre_qrs_max",
    "qrs_window_mean", "qrs_window_std", "qrs_window_min", "qrs_window_max",
    "post_qrs_mean", "post_qrs_std", "post_qrs_min", "post_qrs_max",
] + [f"{c}_{s}" for c in ["cA4", "cD4", "cD3", "cD2", "cD1"]
     for s in ["std", "mean_abs", "energy_ratio", "entropy"]]


def wavelet_features(signal, fs=100):
    """62 features of one lead (SVM_ECG_Wavelet_Lead_Comparison.ipynb, extract_ecg_wavelet_features)."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size < 64:
        raise ValueError(f"Signal is too short for feature extraction: {x.size} samples")

    # band-pass for the measurements; a separate 5-20 Hz copy only for R-peak candidates
    nyquist = fs / 2.0
    high = min(40.0, nyquist * 0.90)
    low = min(0.5, high * 0.25)
    sos = butter(3, [low, high], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, x)
    centered = filtered - np.median(filtered)

    diff = np.diff(centered)
    abs_diff = np.abs(diff)
    std = np.std(centered)
    q10, q25, q75, q90 = np.quantile(centered, [0.10, 0.25, 0.75, 0.90])
    crossings = np.mean((centered[:-1] >= 0) != (centered[1:] >= 0))

    values = [
        np.mean(centered), std, np.sqrt(np.mean(np.square(centered))), np.median(centered),
        np.median(np.abs(centered - np.median(centered))), np.min(centered), np.max(centered),
        np.ptp(centered), q10, q25, q75, q90,
        skew(centered, bias=False), kurtosis(centered, fisher=True, bias=False),
        np.mean(abs_diff), np.std(diff), np.sum(abs_diff) / max(1, len(centered) - 1), crossings,
    ]

    # coarse R-peak candidates on a QRS-emphasized signal (approximate, not clinical measurements)
    det_low, det_high = 5.0, min(20.0, nyquist * 0.90)
    detector = butter(2, [det_low, det_high], btype="bandpass", fs=fs, output="sos")
    qrs_signal = np.abs(sosfiltfilt(detector, centered))
    prominence = max(float(np.std(qrs_signal) * 0.35), 1e-8)
    peaks, peak_properties = find_peaks(qrs_signal, distance=max(1, int(0.30 * fs)), prominence=prominence)

    rr = np.diff(peaks) / float(fs) if len(peaks) >= 2 else np.asarray([], dtype=float)
    if rr.size:
        rr_mean = np.mean(rr)
        rr_std = np.std(rr)
        rr_median = np.median(rr)
        rr_iqr = np.quantile(rr, 0.75) - np.quantile(rr, 0.25)
        rr_min, rr_max = np.min(rr), np.max(rr)
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr)))) if rr.size >= 2 else 0.0
        mean_hr = 60.0 / rr_mean if rr_mean > 0 else 0.0
    else:
        rr_mean = rr_std = rr_median = rr_iqr = rr_min = rr_max = rmssd = mean_hr = 0.0

    peak_amplitudes = qrs_signal[peaks] if len(peaks) else np.asarray([], dtype=float)
    prominences = peak_properties.get("prominences", np.asarray([], dtype=float))
    widths = peak_widths(qrs_signal, peaks, rel_height=0.5)[0] / float(fs) if len(peaks) else np.asarray([], dtype=float)

    def median_or_zero(array):
        return float(np.median(array)) if len(array) else 0.0

    def iqr_or_zero(array):
        return float(np.quantile(array, 0.75) - np.quantile(array, 0.25)) if len(array) else 0.0

    values.extend([
        rr_mean, rr_std, rr_median, rr_iqr, rr_min, rr_max, rmssd, mean_hr,
        median_or_zero(peak_amplitudes), iqr_or_zero(peak_amplitudes),
        median_or_zero(prominences), median_or_zero(widths),
    ])

    # median R-aligned beat, summarized in windows relative to the QRS (not a clinical delineation)
    pre_samples = int(round(0.22 * fs))
    post_samples = int(round(0.38 * fs))
    beat_segments = [centered[p - pre_samples:p + post_samples] for p in peaks
                     if p >= pre_samples and p + post_samples <= len(centered)]
    if beat_segments:
        median_beat = np.median(np.vstack(beat_segments), axis=0)
        beat_times = (np.arange(len(median_beat)) - pre_samples) / float(fs)
        for mask in [(beat_times >= -0.20) & (beat_times < -0.08),
                     (beat_times >= -0.06) & (beat_times <= 0.06),
                     (beat_times >= 0.10) & (beat_times <= 0.30)]:
            segment = median_beat[mask]
            values.extend([np.mean(segment), np.std(segment), np.min(segment), np.max(segment)]
                          if segment.size else [0.0, 0.0, 0.0, 0.0])
    else:
        values.extend([0.0] * 12)

    # discrete wavelet transform, db4, order [cA4, cD4, cD3, cD2, cD1]
    level = min(4, dwt_max_level(len(centered), len(DB4_DEC_LO)))
    coeffs = wavedec(centered, level)
    total_energy = sum(float(np.sum(np.square(c))) for c in coeffs) + 1e-12
    for coeff in coeffs:
        values.extend([np.std(coeff), np.mean(np.abs(coeff)),
                       np.sum(np.square(coeff)) / total_energy, _coefficient_entropy(coeff)])

    if len(values) != len(WAVELET_NAMES):
        raise ValueError(f"Feature length mismatch ({len(values)} vs {len(WAVELET_NAMES)}); "
                         f"DWT level was {level}, expected 4 at {fs} Hz.")
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# ----------------------------------------------------------------------------- extraction + cache

EXTRACTORS = {"statistical": statistical_features, "wavelet": wavelet_features}
FEATURE_NAMES = {"statistical": STATISTICAL_NAMES, "wavelet": WAVELET_NAMES}
VERSIONS = {"statistical": "statistical_v1", "wavelet": "ecg_wavelet_v2"}


def _extract_chunk(kind: str, x: np.ndarray, fs: int) -> np.ndarray:
    fn = EXTRACTORS[kind]
    out = np.empty((x.shape[0], x.shape[1], len(FEATURE_NAMES[kind])), dtype=np.float32)
    for i in range(x.shape[0]):
        for lead in range(x.shape[1]):
            out[i, lead] = fn(x[i, lead], fs)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def extract(x: np.ndarray, kind: str, fs: int, workers: int | None = None, chunk: int = 128) -> np.ndarray:
    """(n, leads, samples) raw mV -> (n, leads, n_features), in parallel over records."""
    if kind not in EXTRACTORS:
        raise ValueError(f"kind must be one of {list(EXTRACTORS)}, got {kind!r}")
    parts = [x[i:i + chunk] for i in range(0, len(x), chunk)]
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        return np.concatenate(list(pool.map(_extract_chunk, [kind] * len(parts), parts, [fs] * len(parts))))


def features_path(kind: str, fs: int):
    return PROCESSED_DIR / f"features_{kind}_{fs}hz.npz"


def load_features(kind: str, fs: int, splits: dict, workers: int | None = None) -> dict[str, np.ndarray]:
    """Per-lead features (n, 12, n_features) for every split of `splits`
    (from load_splits(fs, "all12"), raw mV); computed once and cached."""
    names = ("train", "val", "test")
    ids = np.concatenate([splits[k].ecg_id for k in names])
    path = features_path(kind, fs)
    feats = None
    if path.exists():
        with np.load(path) as d:
            if str(d["version"]) == VERSIONS[kind] and np.array_equal(d["ecg_id"], ids):
                feats = d["features"]
    if feats is None:
        x = np.concatenate([splits[k].x for k in names])
        if x.shape[1] != 12:
            raise ValueError("features are cached for all 12 leads; load the splits with leads='all12'")
        feats = extract(x, kind, fs, workers)
        np.savez_compressed(path, features=feats, ecg_id=ids, version=np.asarray(VERSIONS[kind]),
                            feature_names=np.asarray(FEATURE_NAMES[kind]))
    bounds = np.cumsum([0] + [len(splits[k].ecg_id) for k in names])
    return {k: feats[bounds[i]:bounds[i + 1]] for i, k in enumerate(names)}
