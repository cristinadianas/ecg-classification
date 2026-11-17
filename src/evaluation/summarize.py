"""Collect all finished runs of an experiment into tables (mean ± std over seeds).

    python -m src.evaluation.summarize --exp e1_depth

    runs/<exp>/runs.csv      one row per run (config + every val_/test_ metric)
    runs/<exp>/summary.csv   one row per variant: mean and std over seeds
    runs/<exp>/summary.md    the headline table, ready to paste into the report
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import CLASSES, RUNS_DIR

CONFIG_COLS = ["variant", "run_name", "model", "fs", "leads", "augment", "pos_weight", "seed",
               "n_params", "epochs", "best_epoch", "train_time_s"]
HEADLINE = ["test_macro_auc", "test_macro_ap", "test_macro_f1", "test_tuned_macro_f1",
            *[f"test_auc_{c}" for c in CLASSES]]


def collect_runs(exp_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(p for p in exp_dir.iterdir() if (p / "metrics.json").exists()):
        cfg = json.loads((run_dir / "config.json").read_text())
        m = json.loads((run_dir / "metrics.json").read_text())
        row = {k: cfg.get(k) for k in CONFIG_COLS}
        for split in ("val", "test"):
            row.update({f"{split}_{k}": v for k, v in m[split].items()})
        # last-epoch checkpoint, if evaluated (evaluate --checkpoint last): last_test_* etc.
        if (run_dir / "metrics_last.json").exists():
            ml = json.loads((run_dir / "metrics_last.json").read_text())
            for split in ("val", "test"):
                row.update({f"last_{split}_{k}": v for k, v in ml[split].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in runs.columns
                   if c.startswith(("val_", "test_", "last_val_", "last_test_"))] + ["best_epoch"]
    g = runs.groupby("variant", sort=False)
    agg = g[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    info = g[["model", "fs", "leads", "augment", "n_params"]].first()
    info["n_seeds"] = g.size()
    return info.join(agg).reset_index()


def to_markdown(summary: pd.DataFrame) -> str:
    cols = ["variant", "n_params", "n_seeds", *HEADLINE]
    head = ["variant", "params", "seeds", "macro AUC", "macro AP", "F1@0.5", "F1@tuned",
            *[f"AUC {c}" for c in CLASSES]]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in summary.iterrows():
        params = "–" if pd.isna(r["n_params"]) else f"{r['n_params'] / 1e6:.1f}M"  # "–": not a neural network
        cells = [r["variant"], params, str(r["n_seeds"])]
        for c in HEADLINE:
            std = r[f"{c}_std"]
            cells.append(f"{r[f'{c}_mean']:.4f}" + ("" if pd.isna(std) else f" ± {std:.4f}"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary(exp: str, show: bool = True) -> pd.DataFrame | None:
    """show=False writes the files without printing the (test) table."""
    exp_dir = RUNS_DIR / exp
    runs = collect_runs(exp_dir)
    if runs.empty:
        print(f"no finished runs in {exp_dir}")
        return None
    summary = summarize(runs)
    runs.to_csv(exp_dir / "runs.csv", index=False, float_format="%.6g")
    summary.to_csv(exp_dir / "summary.csv", index=False, float_format="%.6g")
    md = to_markdown(summary)
    (exp_dir / "summary.md").write_text(
        f"# {exp}\n\nTest set = fold 10, checkpoint = best validation macro AUC. "
        f"Mean ± std over seeds.\n\n{md}\n", encoding="utf-8")
    if show:
        print(md)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", required=True)
    write_summary(p.parse_args(argv).exp)


if __name__ == "__main__":
    main()
