"""Train one RBF support vector machine on the official PTB-XL split, then evaluate it.

Classical, non-deep baseline (B0). The model follows the original SVM notebooks in
src/models/ (kept for reference) and is run in the standard pipeline: official split
(train folds 1-8, val 9, test 10; unlabeled records dropped), selection on validation
macro AUROC, and the same run outputs as a neural network, so summaries, bootstrap
CIs, figures and src.evaluation.age_analysis work unchanged.

    python -m src.training.train_svm --exp b0_svm --features wavelet
    python -m src.training.train_svm --exp e4_leads --features statistical --leads lead2

Input (--features):
    statistical  39 statistical/spectral features per lead (src/features/ecg_features.py)
    wavelet      62 ECG + wavelet features per lead
    raw          every sample of the selected leads (1000 per lead at 100 Hz)

Model: inputs z-scored with training statistics; one binary RBF SVM per class
(one-vs-rest, class_weight="balanced"), all classes sharing C and gamma. The RBF kernel
is computed on the GPU in float64 and passed to libsvm as a precomputed kernel: the
same model as SVC(kernel="rbf"), but fast enough for the 12,000-dimensional raw input.
Grid, identical for every input and lead set: C in {0.25, 1, 4, 16, 64} x gamma in
{0.125, 0.25, ..., 8} x gamma_scale, with gamma_scale = 1 / (n_inputs * Var(X_train))
(scikit-learn's gamma="scale"). Selected: highest validation macro AUROC.
Class probabilities = sigmoid(decision function); F1 at 0.5 (decision function 0) and
at per-class thresholds tuned on validation.

The SVM has no random component (SVC.random_state only matters with probability=True),
so --seeds only labels the run: every seed gives the same result.

Run directory runs/<exp>/svm-<features>_<fs>hz_<leads>_s<seed>/:
    config.json      settings, selected C and gamma, validation score of every grid point
    metrics.json     val + test metrics (same keys as src.evaluation.evaluate)
    predictions.npz  val/test ecg_id, y, prob
"""

import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from src.config import CLASSES, ROOT, RUNS_DIR, STANDARD
from src.data.dataset import load_splits, resolve_leads
from src.evaluation.evaluate import resolve_device
from src.evaluation.metrics import bootstrap_ci, compute_metrics, tune_thresholds
from src.features.ecg_features import load_features
from src.training.svm_fit import fit_binary
from src.training.train import git_commit

FEATURES = ("statistical", "wavelet", "raw")
GRID_C = (0.25, 1, 4, 16, 64)
GRID_GAMMA = (0.125, 0.25, 0.5, 1, 2, 4, 8)  # multiples of gamma_scale
SPLITS = ("train", "val", "test")


def default_config(**overrides) -> dict:
    cfg = {"exp": "scratch", "features": "wavelet", "fs": STANDARD["fs"], "leads": STANDARD["leads"],
           "seed": 0, "tag": "", "run_name": None, "overwrite": False, "device": "auto",
           "n_jobs": 20, "cache_mb": 500, "bootstrap": STANDARD["bootstrap"]}
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise ValueError(f"unknown config keys {sorted(unknown)}")
    cfg.update(overrides)
    if cfg["features"] not in FEATURES:
        raise ValueError(f"features must be one of {FEATURES}, got {cfg['features']!r}")
    return cfg


def make_run_name(cfg: dict) -> tuple[str, str]:
    lead_tag, _ = resolve_leads(cfg["leads"])
    parts = [f"svm-{cfg['features']}", f"{cfg['fs']}hz", lead_tag] + ([cfg["tag"]] if cfg["tag"] else [])
    variant = "_".join(parts)
    return cfg["run_name"] or f"{variant}_s{cfg['seed']}", variant


def design_matrices(cfg: dict, splits: dict) -> dict[str, np.ndarray]:
    """Standardized inputs per split (float64), for the selected leads."""
    _, lead_idx = resolve_leads(cfg["leads"])
    if cfg["features"] == "raw":
        x = {k: splits[k].x[:, lead_idx].reshape(len(splits[k].x), -1) for k in SPLITS}
    else:
        f = load_features(cfg["features"], cfg["fs"], splits)
        x = {k: f[k][:, lead_idx].reshape(len(f[k]), -1) for k in SPLITS}
    scaler = StandardScaler().fit(x["train"])
    return {k: scaler.transform(v).astype(np.float64) for k, v in x.items()}


@torch.no_grad()
def squared_distances(a: np.ndarray, b: np.ndarray, device: torch.device) -> np.ndarray:
    ta = torch.as_tensor(a, dtype=torch.float64, device=device)
    tb = ta if b is a else torch.as_tensor(b, dtype=torch.float64, device=device)
    d = (ta * ta).sum(1)[:, None] + (tb * tb).sum(1)[None, :] - 2.0 * (ta @ tb.T)
    return d.clamp_min_(0.0).cpu().numpy()


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def train(cfg: dict):
    cfg = default_config(**cfg)
    name, variant = make_run_name(cfg)
    run_dir = RUNS_DIR / cfg["exp"] / name
    if (run_dir / "metrics.json").exists() and not cfg["overwrite"]:
        print(f"[skip] {run_dir.relative_to(ROOT)} is complete (use --overwrite to redo)")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg["device"])
    started, t_start = datetime.now().isoformat(timespec="seconds"), time.time()

    # ---- inputs and kernel distances
    splits = load_splits(cfg["fs"], "all12")
    x = design_matrices(cfg, splits)
    n_inputs = x["train"].shape[1]
    gamma_scale = 1.0 / (n_inputs * x["train"].var())
    dist = {k: squared_distances(x[k], x["train"], device) for k in SPLITS}
    np.fill_diagonal(dist["train"], 0.0)
    del x
    lead_tag, lead_idx = resolve_leads(cfg["leads"])
    print(f"[run] {run_dir.relative_to(ROOT)}  |  SVM, {cfg['features']}, {n_inputs} inputs  |  "
          f"grid {len(GRID_C)} C x {len(GRID_GAMMA)} gamma  ({time.time() - t_start:.0f}s to prepare)", flush=True)

    # ---- grid: one shared (C, gamma) for the five one-vs-rest SVMs, selected on validation macro AUROC
    grid, best = [], None
    y = {k: splits[k].y for k in SPLITS}
    for gm in GRID_GAMMA:
        gamma = gm * gamma_scale
        k = {s: np.exp(-gamma * dist[s]) for s in SPLITS}
        jobs = [(c, j) for c in GRID_C for j in range(len(CLASSES))]
        t_gamma = time.time()
        # processes, not threads: libsvm holds the GIL while fitting. The kernels reach the
        # workers as read-only memory maps (joblib), so they are not copied per worker.
        out = Parallel(n_jobs=min(cfg["n_jobs"], len(jobs)), backend="loky", mmap_mode="r")(
            delayed(fit_binary)(k["train"], y["train"][:, j].astype(int), k["val"], k["test"], c, cfg["cache_mb"])
            for c, j in jobs)
        res = dict(zip(jobs, out))
        print(f"  gamma={gm} x scale: {len(jobs)} SVMs in {time.time() - t_gamma:.0f}s wall time", flush=True)
        for c in GRID_C:
            val = np.stack([res[(c, j)][0] for j in range(len(CLASSES))], axis=1)
            test = np.stack([res[(c, j)][1] for j in range(len(CLASSES))], axis=1)
            auc = compute_metrics(y["val"], sigmoid(val))["macro_auc"]
            point = {"C": c, "gamma_mult": gm, "gamma": gamma, "val_macro_auc": auc,
                     "n_support": [res[(c, j)][2] for j in range(len(CLASSES))],
                     "fit_s": round(max(res[(c, j)][3] for j in range(len(CLASSES))), 1)}
            grid.append(point)
            print(f"  C={c:<3} gamma={gm} x scale  val macro AUC {auc:.4f}  "
                  f"(slowest class fit {point['fit_s']:.0f}s)", flush=True)
            if best is None or auc > best["point"]["val_macro_auc"]:
                best = {"point": point, "val": val, "test": test}
        del k

    # ---- evaluation of the selected configuration (same format as src.evaluation.evaluate)
    prob = {"val": sigmoid(best["val"]), "test": sigmoid(best["test"])}
    metrics = {"run_name": name, "variant": variant, "checkpoint": "selected", "epoch": None}
    for s in ("val", "test"):
        metrics[s] = compute_metrics(y[s], prob[s])
    thr = tune_thresholds(y["val"], prob["val"])
    metrics["thresholds_tuned"] = dict(zip(CLASSES, thr.tolist()))
    tuned = compute_metrics(y["test"], prob["test"], thr, prefix="tuned_")
    metrics["test"].update({key: v for key, v in tuned.items() if "f1" in key})
    if cfg["bootstrap"]:
        metrics["test"].update(bootstrap_ci(y["test"], prob["test"], n_boot=cfg["bootstrap"]))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(run_dir / "predictions.npz", **{
        f"{s}_{f}": v for s in ("val", "test")
        for f, v in (("ecg_id", splits[s].ecg_id), ("y", y[s]), ("prob", prob[s]))})

    sel = best["point"]
    meta = {**cfg, "run_name": name, "variant": variant, "model": "svm", "augment": "none", "pos_weight": False,
            "lead_tag": lead_tag, "lead_indices": lead_idx, "n_leads": len(lead_idx), "n_inputs": n_inputs,
            "n_params": None, "epochs": None, "best_epoch": None,
            "n_train": len(y["train"]), "n_val": len(y["val"]), "n_test": len(y["test"]),
            "gamma_scale": gamma_scale, "selected": {"C": sel["C"], "gamma_mult": sel["gamma_mult"],
                                                     "gamma": sel["gamma"], "n_support": sel["n_support"]},
            "grid": grid, "train_time_s": time.time() - t_start,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch": torch.__version__, "git_commit": git_commit(), "started_at": started,
            "finished_at": datetime.now().isoformat(timespec="seconds"), "status": "done"}
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2))

    t = metrics["test"]
    ci = f" [{t['macro_auc_ci_low']:.4f}, {t['macro_auc_ci_high']:.4f}]" if "macro_auc_ci_low" in t else ""
    print(f"[test] {run_dir.relative_to(ROOT)} (C={sel['C']}, gamma={sel['gamma_mult']} x scale): "
          f"macro AUC {t['macro_auc']:.4f}{ci}  F1@0.5 {t['macro_f1']:.4f}  F1@tuned {t['tuned_macro_f1']:.4f}  "
          + "  ".join(f"{c} {t['auc_' + c]:.3f}" for c in CLASSES), flush=True)
    return run_dir


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default="scratch", help="experiment name = folder under runs/")
    p.add_argument("--features", default="wavelet", choices=FEATURES)
    p.add_argument("--fs", type=int, default=STANDARD["fs"])
    p.add_argument("--leads", default=STANDARD["leads"], help="preset (all12, limb6, lead2, ...) or lead list")
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="labels only: the SVM is deterministic")
    p.add_argument("--tag", default="")
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default="auto", help="device for the kernel computation")
    p.add_argument("--n-jobs", type=int, default=20, help="binary SVMs trained in parallel (processes)")
    p.add_argument("--cache-mb", type=int, default=500, help="libsvm kernel cache per SVM")
    p.add_argument("--bootstrap", type=int, default=STANDARD["bootstrap"], help="0 disables test CIs")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv=None):
    args = vars(build_parser().parse_args(argv))
    seeds = args.pop("seeds")
    return [train({**args, "seed": s}) for s in seeds]


if __name__ == "__main__":
    main()
