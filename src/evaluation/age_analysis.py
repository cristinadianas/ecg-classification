"""Test metrics per age group and per sex for the runs of an experiment.

    python -m src.evaluation.age_analysis --exp b0_baseline
    python -m src.evaluation.age_analysis --exp e1_depth --variant resnet1d18_100hz_all12
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import CLASSES, RAW_DIR, ROOT, RUNS_DIR
from src.evaluation.metrics import compute_metrics

AGE_PLACEHOLDER, AGE_PLACEHOLDER_AS = 300, 90
DEFAULT_BINS = (0, 40, 60, 80, 200)
SEX_NAMES = {0: "male", 1: "female"}


def load_demographics(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "ptbxl_database.csv", index_col="ecg_id", usecols=["ecg_id", "age", "sex"])
    df["age"] = df["age"].replace(AGE_PLACEHOLDER, AGE_PLACEHOLDER_AS)
    return df


def age_labels(bins) -> list[str]:
    labels = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if lo == 0:
            labels.append(f"<{hi}")
        elif hi >= 120:
            labels.append(f"{lo}+")
        else:
            labels.append(f"{lo}-{hi - 1}")
    return labels


def assign_groups(ecg_ids: np.ndarray, demo: pd.DataFrame, by: str, bins) -> pd.Series:
    d = demo.reindex(ecg_ids)
    if by == "age":
        g = pd.cut(d["age"], bins=list(bins), right=False, labels=age_labels(bins))
    else:
        g = d["sex"].map(SEX_NAMES)
    return pd.Series(np.asarray(g, dtype=object), index=ecg_ids)


def group_order(by: str, bins) -> list[str]:
    return age_labels(bins) if by == "age" else list(SEX_NAMES.values())


def bootstrap_macro_auc(y, p, n_boot: int, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    if n_boot <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n, stats = len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        aucs = [roc_auc_score(yb[:, k], pb[:, k]) for k in range(y.shape[1])
                if 0 < yb[:, k].sum() < n]
        if aucs:
            stats.append(np.mean(aucs))
    if not stats:
        return np.nan, np.nan
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def finished_runs(exp_dir: Path, variant: str | None) -> list[Path]:
    runs = []
    for d in sorted(p for p in exp_dir.iterdir() if (p / "predictions.npz").exists()):
        cfg = json.loads((d / "config.json").read_text())
        if variant is None or cfg["variant"] == variant:
            runs.append(d)
    return runs


def evaluate_groups(run_dir: Path, demo: pd.DataFrame, by: str, bins, n_boot: int) -> list[dict]:
    cfg = json.loads((run_dir / "config.json").read_text())
    pred = np.load(run_dir / "predictions.npz")
    ids, y, p = pred["test_ecg_id"], pred["test_y"].astype(int), pred["test_prob"]
    groups = assign_groups(ids, demo, by, bins).to_numpy()
    rows = []
    for g in ["all", *group_order(by, bins)]:
        mask = np.ones(len(ids), bool) if g == "all" else (groups == g)
        n = int(mask.sum())
        row = {"variant": cfg["variant"], "run_name": cfg["run_name"], "seed": cfg["seed"],
               "group": g, "n": n}
        if n == 0:
            rows.append(row)
            continue
        yg, pg = y[mask], p[mask]
        m = compute_metrics(yg, pg)
        row.update({k: m[k] for k in m if k.startswith(("macro_auc", "macro_ap", "auc_", "ap_"))})
        row.update({f"pos_{c}": int(yg[:, k].sum()) for k, c in enumerate(CLASSES)})
        row["macro_auc_ci_low"], row["macro_auc_ci_high"] = bootstrap_macro_auc(yg, pg, n_boot, cfg["seed"])
        rows.append(row)
    return rows


def composition(run_dir: Path, demo: pd.DataFrame, by: str, bins) -> pd.DataFrame:
    pred = np.load(run_dir / "predictions.npz")
    ids, y = pred["test_ecg_id"], pred["test_y"].astype(int)
    groups = assign_groups(ids, demo, by, bins).to_numpy()
    rows = []
    for g in ["all", *group_order(by, bins), "missing"]:
        mask = (np.ones(len(ids), bool) if g == "all"
                else pd.isna(groups) if g == "missing" else groups == g)
        n = int(mask.sum())
        if g == "missing" and n == 0:
            continue
        row = {"group": g, "n": n, "share_%": 100 * n / len(ids)}
        row.update({f"prev_{c}_%": 100 * y[mask, k].mean() if n else np.nan
                    for k, c in enumerate(CLASSES)})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    metric_cols = [c for c in runs.columns
                   if c.startswith(("macro_", "auc_", "ap_", "pos_")) or c == "n"]
    g = runs.groupby(["variant", "group"], sort=False)
    agg = g[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg["n_seeds"] = g.size()
    out = agg.reset_index()
    out["group"] = pd.Categorical(out["group"], ["all", *order], ordered=True)
    return out.sort_values(["variant", "group"]).reset_index(drop=True)


def to_markdown(summary: pd.DataFrame, comp: pd.DataFrame, by: str, min_pos: int) -> str:
    lines = [f"Test set = fold 10, checkpoint = best validation macro AUC. Mean ± std over seeds; "
             f"CI = 95% bootstrap interval of the macro AUROC (mean of the per-seed bounds). "
             f"Per-class AUROC on fewer than {min_pos} positives is shown as -. "
             f"All metrics: higher is better.", ""]

    lines += [f"## Test-set composition by {by}", ""]
    head = ["group", "n", "% of test", *[f"% {c}" for c in CLASSES]]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in comp.iterrows():
        lines.append("| " + " | ".join([str(r["group"]), str(int(r["n"])), f"{r['share_%']:.1f}",
                                        *[f"{r[f'prev_{c}_%']:.1f}" for c in CLASSES]]) + " |")

    for variant, s in summary.groupby("variant", sort=False):
        lines += ["", f"## {variant}: test AUROC by {by}", ""]
        head = ["group", "n", "macro AUROC", "95% CI", *[f"AUC {c}" for c in CLASSES]]
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for _, r in s.iterrows():
            if pd.isna(r.get("macro_auc_mean")):
                continue
            std = r["macro_auc_std"]
            cells = [str(r["group"]), str(int(r["n_mean"])),
                     f"{r['macro_auc_mean']:.4f}" + ("" if pd.isna(std) else f" ± {std:.4f}"),
                     f"[{r['macro_auc_ci_low_mean']:.3f}, {r['macro_auc_ci_high_mean']:.3f}]"]
            for c in CLASSES:
                few = r[f"pos_{c}_mean"] < min_pos or pd.isna(r[f"auc_{c}_mean"])
                cells.append("-" if few else f"{r[f'auc_{c}_mean']:.3f}")
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _variant_rows(s_all: pd.DataFrame, variant: str, groups: list[str]) -> pd.DataFrame:
    s = s_all[s_all["variant"] == variant]
    return s.set_index(s["group"].astype(str)).reindex(groups)


def plot_groups(summary: pd.DataFrame, by: str, fig_dir: Path, min_pos: int):
    from src.visualize.plot_runs import SERIES

    fig_dir.mkdir(parents=True, exist_ok=True)
    variants = list(dict.fromkeys(summary["variant"]))
    s_all = summary[summary["group"] != "all"].dropna(subset=["macro_auc_mean"])
    groups = list(dict.fromkeys(s_all["group"].astype(str)))
    x = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(1.4 + 1.3 * len(groups), 3.2))
    width = 0.8 / max(len(variants), 1)
    for i, v in enumerate(variants):
        s = _variant_rows(s_all, v, groups)
        mean = s["macro_auc_mean"].to_numpy()
        err = np.vstack([mean - s["macro_auc_ci_low_mean"], s["macro_auc_ci_high_mean"] - mean])
        xi = x + (i - (len(variants) - 1) / 2) * width
        ax.errorbar(xi, mean, yerr=err, fmt="o", color=SERIES[i % len(SERIES)], capsize=3, label=v)
        if i == 0:
            ticks = [f"{g}\nn={int(n)}" for g, n in zip(groups, s["n_mean"])]
    overall = summary[summary["group"] == "all"]
    for i, v in enumerate(variants):
        o = overall[overall["variant"] == v]["macro_auc_mean"]
        if len(o):
            ax.axhline(float(o.iloc[0]), color=SERIES[i % len(SERIES)], lw=0.8, ls="--", alpha=0.6)
    ax.set_xticks(x, ticks)
    ax.set_xlabel(by)
    ax.set_ylabel("test macro AUROC (higher is better)")
    ax.set_title(f"Test macro AUROC by {by}", pad=16)
    ax.text(0, 1.02, "dashed: whole test set  |  bars: 95% bootstrap CI", transform=ax.transAxes,
            fontsize=7, color="#898781", va="bottom")
    if len(variants) > 1:
        ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=len(variants))
    fig.tight_layout()
    fig.savefig(fig_dir / f"{by}_auc.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(variants), figsize=(max(5.5, 4.2 * len(variants)), 3.4), squeeze=False)
    for ax, v in zip(axes[0], variants):
        s = _variant_rows(s_all, v, groups)
        for k, c in enumerate(CLASSES):
            vals = s[f"auc_{c}_mean"].where(s[f"pos_{c}_mean"] >= min_pos).to_numpy()
            ax.plot(x, vals, "o-", color=SERIES[k % len(SERIES)], label=c)
        ax.set_xticks(x, groups)
        ax.set_xlabel(by)
        ax.set_title(v, fontsize=9)
    axes[0][0].set_ylabel("test AUROC (higher is better)")
    axes[0][-1].legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"Per-class test AUROC by {by}\n(points with < {min_pos} positives omitted)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{by}_auc_per_class.png", dpi=200)
    plt.close(fig)


def analyse(exp: str, variant: str | None = None, bins=DEFAULT_BINS, n_boot: int = 1000,
            min_pos: int = 10, raw_dir: Path = RAW_DIR):
    exp_dir = RUNS_DIR / exp
    runs = finished_runs(exp_dir, variant)
    if not runs:
        raise SystemExit(f"no runs with predictions.npz in {exp_dir}"
                         + (f" for variant {variant}" if variant else ""))
    demo = load_demographics(raw_dir)
    out_dir = exp_dir / "subgroups"
    out_dir.mkdir(parents=True, exist_ok=True)

    for by in ("age", "sex"):
        order = group_order(by, bins)
        comp = composition(runs[0], demo, by, bins)
        rows = [r for d in runs for r in evaluate_groups(d, demo, by, bins, n_boot)]
        per_run = pd.DataFrame(rows)
        summary = summarize(per_run, order)

        comp.to_csv(out_dir / f"{by}_composition.csv", index=False, float_format="%.6g")
        per_run.to_csv(out_dir / f"{by}_runs.csv", index=False, float_format="%.6g")
        summary.to_csv(out_dir / f"{by}_summary.csv", index=False, float_format="%.6g")
        md = to_markdown(summary, comp, by, min_pos)
        (out_dir / f"{by}_summary.md").write_text(f"# {exp}: performance by {by}\n\n{md}\n", encoding="utf-8")
        plot_groups(summary, by, out_dir / "figures", min_pos)
        print(f"\n===== {exp}: by {by} =====\n{md}")
    print(f"\nwrote {out_dir.relative_to(ROOT)}/")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", required=True, help="experiment folder under runs/")
    p.add_argument("--variant", default=None, help="only runs of this variant (default: all)")
    p.add_argument("--bins", type=int, nargs="+", default=list(DEFAULT_BINS),
                   help="age bin edges, left-closed (default: 0 40 60 80 200)")
    p.add_argument("--bootstrap", type=int, default=1000, help="resamples for the CIs (0 disables)")
    p.add_argument("--min-pos", type=int, default=10,
                   help="hide per-class AUROC in the markdown table below this many positives")
    a = p.parse_args(argv)
    analyse(a.exp, a.variant, a.bins, a.bootstrap, a.min_pos)


if __name__ == "__main__":
    main()
