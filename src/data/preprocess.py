"""Build the processed PTB-XL cache (run once per sampling rate, from the repo root).

    python -m src.data.preprocess                # 100 Hz  (~1 GB, ~30 s)
    python -m src.data.preprocess --fs 100 500   # both    (+5.2 GB for 500 Hz)

Writes to data/processed/:
    labels.csv          one row per record (ecg_id, patient, fold, file names, labels)
    ptbxl_<fs>hz.npy    float32 (21837, 12, fs*10), raw mV, same row order as labels.csv
    ptbxl_<fs>hz.json   shape, lead order and ecg_id order of the array

Lead subsets, the split, unlabeled-record filtering and standardization are applied
at load time (src.data.dataset.load_splits), so this cache serves every experiment.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CLASSES, DURATION_S, LEADS, PROCESSED_DIR, RAW_DIR, SAMPLING_RATES, TEST_FOLDS, TRAIN_FOLDS,
    VAL_FOLDS,
)
from src.data.dataset import LABELS_FILE, cache_path, load_labels


def build_cache(fs: int, raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR,
                workers: int = 16) -> Path:
    """Read every WFDB record at `fs` Hz into one (n_records, 12, fs*10) float32 array."""
    import wfdb  # only needed here

    if fs not in SAMPLING_RATES:
        raise ValueError(f"fs must be one of {SAMPLING_RATES}, got {fs}")
    processed_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(raw_dir)
    labels.to_csv(processed_dir / LABELS_FILE)

    files = labels.filename_lr if fs == 100 else labels.filename_hr
    n_samples = fs * DURATION_S
    path = cache_path(fs, processed_dir)
    # Write straight into a memory-mapped .npy so the 500 Hz version (5.2 GB) never
    # has to fit in RAM twice.
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                    shape=(len(labels), len(LEADS), n_samples))

    def read(i_file):
        i, f = i_file
        sig, meta = wfdb.rdsamp(str(raw_dir / f))
        if meta["fs"] != fs or sig.shape != (n_samples, len(LEADS)):
            raise ValueError(f"{f}: unexpected fs={meta['fs']} shape={sig.shape}")
        if [n.upper() for n in meta["sig_name"]] != [n.upper() for n in LEADS]:
            raise ValueError(f"{f}: unexpected lead order {meta['sig_name']}")
        if not np.isfinite(sig).all():
            raise ValueError(f"{f}: non-finite samples")
        arr[i] = sig.T.astype(np.float32)
        return i

    with ThreadPoolExecutor(workers) as pool:
        for k, _ in enumerate(pool.map(read, enumerate(files)), 1):
            if k % 2000 == 0 or k == len(files):
                print(f"  {fs} Hz: {k:,}/{len(files):,} records")
    arr.flush()
    del arr

    meta = {"fs": fs, "shape": [len(labels), len(LEADS), n_samples], "leads": LEADS,
            "units": "mV", "layout": "(record, lead, time)",
            "ecg_ids": labels.index.tolist()}
    path.with_suffix(".json").write_text(json.dumps(meta))
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fs", type=int, nargs="+", default=[100], choices=SAMPLING_RATES)
    p.add_argument("--workers", type=int, default=16, help="threads reading WFDB files")
    p.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    args = p.parse_args(argv)

    for fs in args.fs:
        path = cache_path(fs)
        if path.exists() and not args.force:
            print(f"{path} exists, skipping (use --force to rebuild)")
            continue
        t0 = time.time()
        print(f"building {path} from {RAW_DIR}")
        build_cache(fs, workers=args.workers)
        arr = np.load(path, mmap_mode="r")
        print(f"wrote {path}  shape={arr.shape}  {path.stat().st_size / 1e9:.2f} GB  "
              f"({time.time() - t0:.0f}s)")

    labels = pd.read_csv(PROCESSED_DIR / LABELS_FILE, index_col="ecg_id")
    labelled = labels[labels.n_labels > 0]
    print(f"\n{len(labels):,} records, {len(labels) - len(labelled)} without a superclass label "
          f"(dropped at load time)")
    for name, folds in (("train", TRAIN_FOLDS), ("val", VAL_FOLDS), ("test", TEST_FOLDS)):
        part = labelled[labelled.strat_fold.isin(folds)]
        counts = "  ".join(f"{c} {int(part[c].sum()):5d}" for c in CLASSES)
        print(f"  {name:5s} folds {str(list(folds)):26s} {len(part):6,} records   {counts}")


if __name__ == "__main__":
    main()
