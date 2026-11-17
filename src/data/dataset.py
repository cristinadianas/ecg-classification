"""PTB-XL loading: label aggregation, the processed cache, lead selection, splits, scaling.

The cache (built by src.data.preprocess) stores *all* records and *all* 12 leads as
raw millivolts. Everything that varies between experiments is applied here, at load
time, so one cache serves all of them:

    load_splits(fs, leads)  ->  lead subset, official split, unlabeled records dropped
    Standardizer            ->  per-lead mean/std fitted on the training folds only
"""

import ast
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CLASSES, LEAD_SETS, LEADS, PROCESSED_DIR, RAW_DIR, TEST_FOLDS, TRAIN_FOLDS, VAL_FOLDS,
)

LABELS_FILE = "labels.csv"


# ----------------------------------------------------------------------------- labels

def load_labels(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Metadata + multi-hot superclass labels for every record, indexed by ecg_id.

    Same aggregation as the PhysioNet example, the Strodthoff et al. benchmark and
    our EDA: every diagnostic SCP statement maps to its superclass, whatever its
    likelihood. Records with no diagnostic statement get all-zero labels (n_labels 0).
    """
    df = pd.read_csv(raw_dir / "ptbxl_database.csv", index_col="ecg_id").sort_index()
    codes = df.scp_codes.apply(ast.literal_eval)
    scp = pd.read_csv(raw_dir / "scp_statements.csv", index_col=0)
    diag = scp[scp.diagnostic == 1].diagnostic_class
    supers = codes.apply(lambda c: {diag[k] for k in c if k in diag.index})
    out = df[["patient_id", "strat_fold", "filename_lr", "filename_hr"]].copy()
    out["patient_id"] = out.patient_id.astype(int)
    for c in CLASSES:
        out[c] = supers.apply(lambda s, c=c: int(c in s))
    out["n_labels"] = out[CLASSES].sum(axis=1)
    return out


def cache_path(fs: int, processed_dir: Path = PROCESSED_DIR) -> Path:
    return processed_dir / f"ptbxl_{fs}hz.npy"


# ----------------------------------------------------------------------------- leads

def resolve_leads(spec: str) -> tuple[str, list[int]]:
    """Lead spec -> (short name used in run names, indices into LEADS).

    `spec` is a preset from LEAD_SETS ("all12", "limb6", "lead2", ...) or a
    comma-separated list of lead names ("I,II,V1"; case-insensitive).
    """
    if spec in LEAD_SETS:
        names, tag = LEAD_SETS[spec], spec
    else:
        lookup = {l.lower(): l for l in LEADS}
        try:
            names = [lookup[s.strip().lower()] for s in spec.split(",") if s.strip()]
        except KeyError as e:
            raise ValueError(f"unknown lead {e.args[0]!r}; leads are {LEADS} "
                             f"and presets are {list(LEAD_SETS)}") from None
        if not names or len(set(names)) != len(names):
            raise ValueError(f"bad lead spec {spec!r}")
        tag = "-".join(names)
    return tag, [LEADS.index(n) for n in names]


# ----------------------------------------------------------------------------- splits

@dataclass
class Split:
    x: np.ndarray        # float32 (n, n_leads, n_samples), raw mV until standardized
    y: np.ndarray        # float32 (n, n_classes), multi-hot
    ecg_id: np.ndarray   # int64 (n,)


def load_splits(fs: int, leads: str = "all12", processed_dir: Path = PROCESSED_DIR,
                drop_unlabeled: bool = True) -> dict[str, Split]:
    """Official split (train folds 1-8, val 9, test 10) for the given rate and leads.

    Records without any superclass label (407 of 21,837) are dropped, as in the
    benchmark, because a multi-label target of all zeros is not a diagnosis.
    """
    path = cache_path(fs, processed_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run: python -m src.data.preprocess --fs {fs}")
    labels = pd.read_csv(processed_dir / LABELS_FILE, index_col="ecg_id")
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta["ecg_ids"] != labels.index.tolist():
        raise RuntimeError("labels.csv and the signal cache disagree; re-run src.data.preprocess")

    _, lead_idx = resolve_leads(leads)
    signals = np.load(path, mmap_mode="r")
    keep = labels.n_labels > 0 if drop_unlabeled else pd.Series(True, index=labels.index)

    splits = {}
    for name, folds in (("train", TRAIN_FOLDS), ("val", VAL_FOLDS), ("test", TEST_FOLDS)):
        rows = np.flatnonzero((labels.strat_fold.isin(folds) & keep).to_numpy())
        x = np.ascontiguousarray(signals[rows][:, lead_idx], dtype=np.float32)
        y = labels.iloc[rows][CLASSES].to_numpy(np.float32)
        splits[name] = Split(x, y, labels.index.to_numpy()[rows])
    return splits


class Standardizer:
    """z-scoring with statistics from the training split only.

    mode "per_lead" (our standard): one mean/std per lead.
    mode "global": one mean/std over all leads and samples, as in the PTB-XL benchmark
    (Strodthoff et al., 2021). Stored as per-lead arrays either way, so transform/load
    do not depend on the mode.
    """

    MODES = ("per_lead", "global")

    def __init__(self, mean=None, std=None, mode: str = "per_lead"):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mean, self.std, self.mode = mean, std, mode

    def fit(self, x: np.ndarray) -> "Standardizer":
        if self.mode == "per_lead":
            self.mean = x.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
            self.std = x.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
        else:
            n_leads = x.shape[1]
            self.mean = np.full(n_leads, x.mean(dtype=np.float64), dtype=np.float32)
            self.std = np.full(n_leads, x.std(dtype=np.float64), dtype=np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """In place, to avoid a second copy of a multi-GB array."""
        x -= self.mean[None, :, None]
        x /= self.std[None, :, None] + 1e-8
        return x

    def save(self, path: Path):
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> "Standardizer":
        d = np.load(path)
        return cls(d["mean"], d["std"])
