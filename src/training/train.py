"""Train one model on the official PTB-XL split, then evaluate it on the test fold.

Defaults are the standard protocol (src/config.py STANDARD); only override what the
experiment is about. Run from the repo root:

    python -m src.training.train --exp e1_depth --model resnet1d34 --seeds 1
    python -m src.training.train --exp e4_leads --leads limb6
    python -m src.training.train --exp e4_leads --leads I,II,V2
    python -m src.training.train --exp e5_fs    --fs 500
    python -m src.training.train --exp e3_aug   --augment noise,wander
    python -m src.training.train --seeds 0 1 2              # several seeds in a row

One run = one model, one input configuration, one seed. Everything it produces goes
to runs/<exp>/<model>_<fs>hz_<leads>[_aug-...]_s<seed>/:

    config.json        every setting + resolved info (params, sizes, git commit, timing)
    norm_stats.npz     per-lead mean/std fitted on the training folds
    history.csv        one row per epoch: lr, losses, macro & per-class AUC/AP/F1
    best.pt / last.pt  weights at the best-validation epoch / after the last epoch
    metrics.json       val + test metrics of best.pt (src.evaluation.evaluate)
    predictions.npz    val/test ecg_ids, labels and probabilities (for ROC/PR plots)

history.csv columns: train_* are computed on the predictions made *during* the epoch
(augmented batches, model still changing, BN in train mode); val_* are computed after
the epoch in eval mode. Use val_* to compare models.
"""

import argparse
import json
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.config import CLASSES, LEAD_SETS, ROOT, RUNS_DIR, SAMPLING_RATES, STANDARD
from src.data.augment import AVAILABLE as AUGMENTATIONS
from src.data.augment import Augmenter, eval_window, parse_augment
from src.data.dataset import Standardizer, load_splits, resolve_leads
from src.evaluation.evaluate import autocast, evaluate_run, predict_logits, resolve_device
from src.evaluation.metrics import compute_metrics
from src.models import MODELS, build_model, count_parameters

GPU_DATA_LIMIT_BYTES = 6 * 1024**3  # keep training data on the GPU when smaller than this


def default_config(**overrides) -> dict:
    cfg = {
        "exp": "scratch", "model": "resnet1d18", "model_kwargs": {}, "seed": 0, "tag": "",
        "pos_weight": False, "device": "auto", "amp": True, "run_name": None,
        "overwrite": False, "bootstrap": STANDARD["bootstrap"],
        **{k: STANDARD[k] for k in ("fs", "leads", "augment", "standardize", "epochs",
                                     "batch_size", "lr", "weight_decay", "adam_beta2",
                                     "pct_start", "dropout")},
    }
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise ValueError(f"unknown config keys {sorted(unknown)}")
    cfg.update(overrides)
    return cfg


def make_run_name(cfg: dict) -> tuple[str, str]:
    """(run_name, variant). The variant is the run name without the seed, used to
    group seeds when summarizing."""
    lead_tag, _ = resolve_leads(cfg["leads"])
    parts = [cfg["model"], f"{cfg['fs']}hz", lead_tag]
    aug = parse_augment(cfg["augment"])
    if aug:
        parts.append("aug-" + "+".join(aug))
    if cfg["pos_weight"]:
        parts.append("posw")
    if cfg["standardize"] != STANDARD["standardize"]:
        parts.append(f"std-{cfg['standardize']}")
    if cfg["tag"]:
        parts.append(cfg["tag"])
    variant = "_".join(parts)
    return cfg["run_name"] or f"{variant}_s{cfg['seed']}", variant


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") if out.returncode == 0 else None
    except Exception:
        return None


def train(cfg: dict) -> Path:
    cfg = default_config(**cfg)
    name, variant = make_run_name(cfg)
    run_dir = RUNS_DIR / cfg["exp"] / name
    if (run_dir / "metrics.json").exists() and not cfg["overwrite"]:
        print(f"[skip] {run_dir.relative_to(ROOT)} is complete (use --overwrite to redo)")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("history.csv", "best.pt", "last.pt", "metrics.json", "predictions.npz"):
        (run_dir / stale).unlink(missing_ok=True)

    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    amp = cfg["amp"] and device.type == "cuda"

    # ---- data
    t0 = time.time()
    splits = load_splits(cfg["fs"], cfg["leads"])
    scaler = Standardizer(mode=cfg["standardize"]).fit(splits["train"].x)
    for s in splits.values():
        scaler.transform(s.x)
    scaler.save(run_dir / "norm_stats.npz")
    tensors = {k: (torch.from_numpy(s.x), torch.from_numpy(s.y)) for k, s in splits.items()}
    x_tr, y_tr = tensors["train"]
    if device.type == "cuda" and x_tr.nbytes < GPU_DATA_LIMIT_BYTES:
        x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_va, y_va = tensors["val"]
    print(f"[data] {cfg['fs']} Hz, leads={cfg['leads']}: train {tuple(x_tr.shape)}, "
          f"val {tuple(x_va.shape)}, test {tuple(tensors['test'][0].shape)} "
          f"({time.time() - t0:.0f}s)")

    # ---- model and optimisation
    n_leads = x_tr.shape[1]
    model = build_model(cfg["model"], n_leads, len(CLASSES), dropout=cfg["dropout"],
                        **cfg["model_kwargs"]).to(device)
    n_params = count_parameters(model)
    steps_per_epoch = len(x_tr) // cfg["batch_size"]  # drop the last partial batch
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                                  betas=(0.9, cfg["adam_beta2"]))  # beta1 is cycled by OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"], total_steps=cfg["epochs"] * steps_per_epoch,
        pct_start=cfg["pct_start"], anneal_strategy="cos")
    pos_weight = None
    if cfg["pos_weight"]:
        p = splits["train"].y.mean(0)
        pos_weight = torch.tensor((1 - p) / p, dtype=torch.float32, device=device)
    augmenter = Augmenter(cfg["augment"], cfg["fs"],
                          torch.Generator(device=device).manual_seed(cfg["seed"]))
    perm_gen = torch.Generator().manual_seed(cfg["seed"])
    window = eval_window(cfg["augment"], cfg["fs"])  # None unless trained on random crops

    lead_tag, lead_idx = resolve_leads(cfg["leads"])
    meta = {**cfg, "run_name": name, "variant": variant, "lead_tag": lead_tag,
            "lead_indices": lead_idx, "n_leads": n_leads, "n_params": n_params,
            "n_train": len(splits["train"].y), "n_val": len(splits["val"].y),
            "n_test": len(splits["test"].y), "steps_per_epoch": steps_per_epoch,
            "eval_window": window, "eval_aggregate": "max" if window else None,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch": torch.__version__, "git_commit": git_commit(),
            "started_at": datetime.now().isoformat(timespec="seconds"), "status": "running"}
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2))
    print(f"[run] {run_dir.relative_to(ROOT)}  |  {cfg['model']} {n_params / 1e6:.2f}M params  |  "
          f"{cfg['epochs']} epochs x {steps_per_epoch} steps  |  {meta['device']}")

    # ---- loop
    history, best_auc, best_epoch = [], -np.inf, 0
    for epoch in range(1, cfg["epochs"] + 1):
        t_ep = time.time()
        model.train()
        perm = torch.randperm(len(x_tr), generator=perm_gen)[: steps_per_epoch * cfg["batch_size"]]
        perm = perm.to(x_tr.device)
        loss_sum, train_logits, train_targets = 0.0, [], []
        for s in range(steps_per_epoch):
            idx = perm[s * cfg["batch_size"]:(s + 1) * cfg["batch_size"]]
            xb = x_tr[idx].to(device, non_blocking=True)
            yb = y_tr[idx].to(device, non_blocking=True)
            if augmenter:
                xb = augmenter(xb)
            with autocast(device, amp):
                logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits.float(), yb, pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(idx)
            train_logits.append(logits.detach().float())
            train_targets.append(yb)
        lr = optimizer.param_groups[0]["lr"]

        tr_prob = torch.sigmoid(torch.cat(train_logits)).cpu().numpy()
        tr_true = torch.cat(train_targets).cpu().numpy()
        va_logits = predict_logits(model, x_va, device, amp, window=window)
        va_loss = F.binary_cross_entropy_with_logits(va_logits.cpu(), y_va).item()
        va_prob = torch.sigmoid(va_logits).cpu().numpy()

        row = {"epoch": epoch, "lr": lr, "time_s": 0.0,
               "train_loss": loss_sum / (steps_per_epoch * cfg["batch_size"]),
               **compute_metrics(tr_true, tr_prob, prefix="train_"),
               "val_loss": va_loss,
               **compute_metrics(splits["val"].y, va_prob, prefix="val_")}
        improved = row["val_macro_auc"] > best_auc
        if improved:
            best_auc, best_epoch = row["val_macro_auc"], epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_macro_auc": best_auc}, run_dir / "best.pt")
        row["time_s"] = time.time() - t_ep
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, float_format="%.6g")
        print(f"  ep {epoch:3d}/{cfg['epochs']}  lr {lr:.1e}  loss {row['train_loss']:.4f}/"
              f"{va_loss:.4f}  val auc {row['val_macro_auc']:.4f}  f1 {row['val_macro_f1']:.4f}  "
              f"{row['time_s']:.1f}s{'  *' if improved else ''}", flush=True)

    torch.save({"model": model.state_dict(), "epoch": cfg["epochs"],
                "val_macro_auc": history[-1]["val_macro_auc"]}, run_dir / "last.pt")
    meta.update(status="trained", best_epoch=best_epoch, best_val_macro_auc=best_auc,
                train_time_s=sum(h["time_s"] for h in history),
                finished_at=datetime.now().isoformat(timespec="seconds"))
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2))

    # ---- test-set evaluation of the selected checkpoint
    evaluate_run(run_dir, "best", splits=splits, device=cfg["device"])
    return run_dir


# ----------------------------------------------------------------------------- CLI

def build_parser():
    S = STANDARD
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default="scratch", help="experiment name = folder under runs/")
    p.add_argument("--model", default="resnet1d18", choices=sorted(MODELS))
    p.add_argument("--model-kwargs", type=json.loads, default={},
                   help='extra model arguments as JSON, e.g. \'{"kernel_size": 7}\'')
    p.add_argument("--fs", type=int, default=S["fs"], choices=SAMPLING_RATES)
    p.add_argument("--leads", default=S["leads"],
                   help=f"preset {list(LEAD_SETS)} or comma-separated lead names")
    p.add_argument("--augment", default=S["augment"],
                   help=f"'none' or comma-separated subset of {list(AUGMENTATIONS)}")
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="one run per seed")
    p.add_argument("--epochs", type=int, default=S["epochs"])
    p.add_argument("--batch-size", type=int, default=S["batch_size"])
    p.add_argument("--lr", type=float, default=S["lr"])
    p.add_argument("--weight-decay", type=float, default=S["weight_decay"])
    p.add_argument("--adam-beta2", type=float, default=S["adam_beta2"])
    p.add_argument("--pct-start", type=float, default=S["pct_start"])
    p.add_argument("--standardize", default=S["standardize"], choices=Standardizer.MODES,
                   help="per-lead z-scoring (standard) or one global mean/std (benchmark)")
    p.add_argument("--dropout", type=float, default=S["dropout"])
    p.add_argument("--pos-weight", action="store_true",
                   help="weight positives by (1-p)/p in the BCE loss (class-imbalance experiments)")
    p.add_argument("--tag", default="", help="suffix to tell apart runs with non-standard settings")
    p.add_argument("--run-name", default=None, help="override the generated run name")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="disable bf16 autocast")
    p.add_argument("--bootstrap", type=int, default=S["bootstrap"], help="0 disables test CIs")
    p.add_argument("--overwrite", action="store_true", help="retrain even if the run is complete")
    return p


def main(argv=None):
    args = vars(build_parser().parse_args(argv))
    seeds = args.pop("seeds")
    if args["run_name"] and len(seeds) > 1:
        raise SystemExit("--run-name only makes sense with a single seed")
    return [train({**args, "seed": seed}) for seed in seeds]


if __name__ == "__main__":
    main()
