"""Evaluate trained runs on the official validation (fold 9) and test (fold 10) sets.

Training already calls evaluate_run at the end of every run; use this command to
re-evaluate (e.g. after changing metrics) or to look at the last-epoch checkpoint.

    python -m src.evaluation.evaluate --exp e1_depth                    # all runs + summary
    python -m src.evaluation.evaluate --run runs/e1_depth/resnet1d18_100hz_all12_s0
    python -m src.evaluation.evaluate --exp e1_depth --checkpoint last  # -> metrics_last.json

Writes to the run directory:
    metrics.json     {"val": {...}, "test": {...}, "thresholds_tuned": {...}, ...}
    predictions.npz  val_/test_ ecg_id, y, prob  (for ROC/PR curves, confusion matrices)

Test metrics:
    macro_auc, auc_<C>, macro_ap, ap_<C>       threshold-free
    macro_f1, f1_<C>                           threshold 0.5
    tuned_macro_f1, tuned_f1_<C>               per-class thresholds that maximise F1 on val
    macro_{auc,ap,f1}_ci_{low,high}            95% bootstrap CIs (records resampled)
    loss                                       BCE
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import CLASSES, ROOT, RUNS_DIR, STANDARD
from src.data.augment import EVAL_STRIDE_FRACTION, eval_window
from src.data.dataset import Standardizer, load_splits
from src.evaluation.metrics import bootstrap_ci, compute_metrics, tune_thresholds
from src.evaluation.summarize import write_summary
from src.models import build_model


# ----------------------------------------------------------------------------- inference

def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def autocast(device: torch.device, enabled: bool):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled and device.type == "cuda")


@torch.no_grad()
def predict_logits(model, x: torch.Tensor, device: torch.device, amp: bool,
                   batch_size: int = 512, window: int | None = None) -> torch.Tensor:
    """One row of class logits per record.

    window: None -> the model sees the full record. Otherwise (models trained with
    random crops) the record is cut into sliding windows of `window` samples with
    stride window * EVAL_STRIDE_FRACTION, and each class takes the maximum over the
    windows (as in the benchmark; max of logits = max of probabilities).
    """
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size].to(device, non_blocking=True)
        if window is None or window >= xb.shape[-1]:
            with autocast(device, amp):
                out.append(model(xb).float())
            continue
        stride = max(1, int(window * EVAL_STRIDE_FRACTION))
        w = xb.unfold(2, window, stride)                      # (B, leads, n_windows, window)
        b, c, n_win, length = w.shape
        w = w.permute(0, 2, 1, 3).reshape(b * n_win, c, length)
        with autocast(device, amp):
            logits = model(w).float().view(b, n_win, -1)
        out.append(logits.amax(dim=1))
    return torch.cat(out)


def load_model(run_dir: Path, checkpoint: str, device: torch.device):
    cfg = json.loads((run_dir / "config.json").read_text())
    model = build_model(cfg["model"], cfg["n_leads"], len(CLASSES), dropout=cfg["dropout"],
                        **cfg.get("model_kwargs", {}))
    ckpt = torch.load(run_dir / f"{checkpoint}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    return cfg, model.to(device), ckpt["epoch"]


# ----------------------------------------------------------------------------- evaluation

def evaluate_run(run_dir: Path, checkpoint: str = "best", splits=None, device: str = "auto",
                 n_boot: int | None = None) -> dict:
    """`splits`: already-standardized splits (passed by training to avoid reloading)."""
    run_dir = Path(run_dir).resolve()
    dev = resolve_device(device)
    cfg, model, epoch = load_model(run_dir, checkpoint, dev)
    amp = cfg.get("amp", True) and dev.type == "cuda"
    n_boot = cfg.get("bootstrap", STANDARD["bootstrap"]) if n_boot is None else n_boot

    if splits is None:
        splits = load_splits(cfg["fs"], cfg["leads"])
        scaler = Standardizer.load(run_dir / "norm_stats.npz")
        for k in ("val", "test"):
            scaler.transform(splits[k].x)

    out = {"run_name": cfg["run_name"], "variant": cfg["variant"], "checkpoint": checkpoint,
           "epoch": epoch}
    probs = {}
    window = eval_window(cfg.get("augment", "none"), cfg["fs"])
    for k in ("val", "test"):
        s = splits[k]
        logits = predict_logits(model, torch.from_numpy(s.x), dev, amp, window=window).cpu()
        probs[k] = torch.sigmoid(logits).numpy()
        out[k] = {"loss": F.binary_cross_entropy_with_logits(logits, torch.from_numpy(s.y)).item(),
                  **compute_metrics(s.y, probs[k])}

    thr = tune_thresholds(splits["val"].y, probs["val"])
    out["thresholds_tuned"] = dict(zip(CLASSES, thr.tolist()))
    tuned = compute_metrics(splits["test"].y, probs["test"], thr, prefix="tuned_")
    out["test"].update({k: v for k, v in tuned.items() if "f1" in k})
    if n_boot:
        out["test"].update(bootstrap_ci(splits["test"].y, probs["test"], n_boot=n_boot))

    if checkpoint == "best":  # the canonical result of the run
        (run_dir / "metrics.json").write_text(json.dumps(out, indent=2))
        np.savez_compressed(
            run_dir / "predictions.npz",
            **{f"{k}_{f}": v for k in ("val", "test")
               for f, v in (("ecg_id", splits[k].ecg_id), ("y", splits[k].y), ("prob", probs[k]))})
        cfg["status"] = "done"
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    else:
        (run_dir / f"metrics_{checkpoint}.json").write_text(json.dumps(out, indent=2))

    t = out["test"]
    ci = (f" [{t['macro_auc_ci_low']:.4f}, {t['macro_auc_ci_high']:.4f}]"
          if "macro_auc_ci_low" in t else "")
    print(f"[test] {run_dir.relative_to(ROOT)} ({checkpoint}, epoch {epoch}): "
          f"macro AUC {t['macro_auc']:.4f}{ci}  F1@0.5 {t['macro_f1']:.4f}  "
          f"F1@tuned {t['tuned_macro_f1']:.4f}  "
          + "  ".join(f"{c} {t['auc_' + c]:.3f}" for c in CLASSES))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--exp", help="evaluate every trained run in runs/<exp>/")
    g.add_argument("--run", nargs="+", type=Path, help="run directories")
    p.add_argument("--checkpoint", default="best", choices=["best", "last"])
    p.add_argument("--bootstrap", type=int, default=None, help="resamples (default: from config)")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    runs = args.run or sorted(d for d in (RUNS_DIR / args.exp).iterdir()
                              if (d / f"{args.checkpoint}.pt").exists())
    for run_dir in runs:
        evaluate_run(run_dir, args.checkpoint, device=args.device, n_boot=args.bootstrap)
    if args.exp and args.checkpoint == "best":
        write_summary(args.exp)


if __name__ == "__main__":
    main()
