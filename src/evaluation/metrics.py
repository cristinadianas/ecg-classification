"""Multi-label metrics, returned as flat dicts so they map 1:1 onto CSV columns.

Keys produced by compute_metrics (prefix omitted):
    macro_auc, auc_<CLASS>     ROC AUC (threshold-free; the benchmark's main metric)
    macro_ap,  ap_<CLASS>      average precision = area under the PR curve
    macro_f1,  f1_<CLASS>      F1 at the given threshold(s)
A per-class value is NaN when that class has only one label value in y_true; macro
averages ignore NaNs.
"""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.config import CLASSES


def _per_class(fn, y_true, y_score):
    out = []
    for k in range(y_true.shape[1]):
        t = y_true[:, k]
        out.append(fn(t, y_score[:, k]) if 0 < t.sum() < len(t) else np.nan)
    return np.array(out, dtype=np.float64)


def per_class_auc(y_true, y_prob):
    return _per_class(roc_auc_score, y_true, y_prob)


def per_class_f1(y_true, y_prob, thresholds):
    """F1 per class; 0 when a class is never predicted and never present."""
    y_pred = y_prob >= np.broadcast_to(thresholds, (y_prob.shape[1],))
    pos = y_true == 1
    tp = (y_pred & pos).sum(0)
    denom = 2 * tp + (y_pred & ~pos).sum(0) + (~y_pred & pos).sum(0)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def compute_metrics(y_true, y_prob, thresholds=0.5, prefix: str = "", classes=CLASSES) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    auc = per_class_auc(y_true, y_prob)
    ap = _per_class(average_precision_score, y_true, y_prob)
    f1 = per_class_f1(y_true, y_prob, thresholds)
    m = {f"{prefix}macro_auc": np.nanmean(auc), f"{prefix}macro_ap": np.nanmean(ap),
         f"{prefix}macro_f1": f1.mean()}
    for name, values in (("auc", auc), ("ap", ap), ("f1", f1)):
        m.update({f"{prefix}{name}_{c}": v for c, v in zip(classes, values)})
    return {k: float(v) for k, v in m.items()}


def tune_thresholds(y_true, y_prob) -> np.ndarray:
    """Per-class threshold that maximises F1. Fit on validation, apply to test."""
    thr = np.full(y_true.shape[1], 0.5)
    for k in range(y_true.shape[1]):
        p, r, t = precision_recall_curve(y_true[:, k], y_prob[:, k])
        f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
        if len(t):
            thr[k] = t[int(np.argmax(f1))]
    return thr


def bootstrap_ci(y_true, y_prob, thresholds=0.5, n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> dict:
    """Percentile bootstrap CIs (resampling records) for macro AUC, AP and F1."""
    y_true = np.asarray(y_true).astype(int)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = {"macro_auc": [], "macro_ap": [], "macro_f1": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = compute_metrics(y_true[idx], y_prob[idx], thresholds)
        for k in stats:
            stats[k].append(m[k])
    out = {}
    for k, v in stats.items():
        lo, hi = np.nanpercentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[f"{k}_ci_low"], out[f"{k}_ci_high"] = float(lo), float(hi)
    return out
