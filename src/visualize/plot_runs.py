"""Plots for any experiment folder under runs/ (mean ± std over seeds per variant).

    python -m src.visualize.plot_runs --exp e1_depth

Writes runs/<exp>/figures/:
    curves.png               train/val loss, val macro AUC, val macro F1 per epoch
    val_auc_per_class.png    val AUC per class per epoch
    test_auc_per_class.png   test AUC (macro + per class) of the selected checkpoints
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.config import CLASSES, RUNS_DIR
from src.evaluation.summarize import collect_runs

# Categorical series colours in fixed order (same palette as the EDA figures).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#8a5cd6", "#c43c3c"]
INK, INK_2, MUTED, GRID = "#14213d", "#4a5568", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.family": ["Calibri", "Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "text.color": INK,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "savefig.dpi": 220,
})


def load_histories(exp_dir: Path) -> pd.DataFrame:
    frames = []
    for run_dir in sorted(exp_dir.iterdir()):
        if not (run_dir / "history.csv").exists():
            continue
        cfg = json.loads((run_dir / "config.json").read_text())
        h = pd.read_csv(run_dir / "history.csv")
        h["variant"], h["seed"] = cfg["variant"], cfg["seed"]
        frames.append(h)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def short_labels(variants: list[str]) -> tuple[dict, str]:
    """Drop the name parts all variants share; return (labels, shared parts)."""
    parts = [v.split("_") for v in variants]
    common = [p for p in parts[0] if all(p in q for q in parts[1:])] if len(parts) > 1 else []
    labels = {v: "_".join(x for x in p if x not in common) or v for v, p in zip(variants, parts)}
    return labels, ", ".join(common)


def band(ax, h: pd.DataFrame, col: str, color: str, label=None, **kw):
    """Mean over seeds with a ±1 std band (band only when there are >= 2 seeds)."""
    g = h.groupby("epoch")[col]
    mean, std = g.mean(), g.std()
    ax.plot(mean.index, mean.values, color=color, label=label, **kw)
    if g.count().min() > 1:
        ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
    return mean


def variant_legend(fig, handles, top: float):
    """Colour legend for the whole figure: one row at the top right when the labels are
    few and short, otherwise below the panels (fewer columns for longer labels) so it
    never meets the subtitle."""
    labels = [h.get_label() for h in handles]
    if len(handles) <= 6 and sum(map(len, labels)) <= 80:
        fig.legend(handles=handles, loc="upper right", ncol=len(handles), fontsize=8,
                   bbox_to_anchor=(1.0, top))
    else:
        longest = max(map(len, labels))
        ncol = 4 if longest <= 28 else 3 if longest <= 40 else 2
        fig.legend(handles=handles, loc="upper center", ncol=ncol, fontsize=8,
                   bbox_to_anchor=(0.5, 0.0))


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", path)


def plot_experiment(exp: str, variant_order: list[str] | None = None,
                    labels: dict[str, str] | None = None):
    """`variant_order` fixes colour and legend order; `labels` maps variant -> legend text
    (default: the variant name without the parts all variants share)."""
    exp_dir = RUNS_DIR / exp
    hist = load_histories(exp_dir)
    if hist.empty:
        print(f"nothing to plot in {exp_dir}")
        return
    variants = variant_order or sorted(hist.variant.unique())
    colors = {v: SERIES[i % len(SERIES)] for i, v in enumerate(variants)}
    auto_labels, shared = short_labels(variants)
    labels = {**auto_labels, **(labels or {})}
    n_seeds = hist.groupby("variant").seed.nunique()
    subtitle = exp + (f"  ·  {shared}" if shared else "")
    if n_seeds.max() > 1:
        seeds = n_seeds.max() if n_seeds.min() == n_seeds.max() else f"{n_seeds.min()}–{n_seeds.max()}"
        subtitle += f"  ·  mean ± std over {seeds} seeds"
    out = exp_dir / "figures"

    # ---- learning curves
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for v in variants:
        h = hist[hist.variant == v]
        band(axes[0], h, "val_loss", colors[v], label=labels[v])
        band(axes[0], h, "train_loss", colors[v], linestyle="--", linewidth=1.4)
        auc = band(axes[1], h, "val_macro_auc", colors[v])
        best = auc.idxmax()
        axes[1].plot(best, auc[best], "o", color=colors[v], markersize=7,
                     markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        band(axes[2], h, "val_macro_f1", colors[v])
    axes[0].set(title="(a) Loss (BCE)", xlabel="epoch", ylabel="loss")
    # line-style key here; one colour legend for the whole figure
    axes[0].legend(handles=[Line2D([], [], color=INK_2, label="validation"),
                            Line2D([], [], color=INK_2, linestyle="--", linewidth=1.4, label="train")],
                   fontsize=7.5, loc="upper left")
    axes[1].set(title="(b) Validation macro AUROC", xlabel="epoch")
    lo = hist.groupby(["variant", "epoch"]).val_macro_auc.mean().quantile(0.1)
    axes[1].set_ylim(max(lo - 0.01, 0.5), None)
    axes[1].annotate("dot = best epoch (mean curve)", xy=(0.98, 0.04), xycoords="axes fraction",
                     ha="right", fontsize=7.5, color=MUTED)
    axes[2].set(title="(c) Validation macro F1 @ 0.5", xlabel="epoch")
    handles = [Line2D([], [], color=colors[v], label=labels[v]) for v in variants]
    variant_legend(fig, handles, top=1.08)
    fig.suptitle(subtitle, x=0.01, ha="left", fontsize=8, color=MUTED, y=1.06)
    fig.tight_layout()
    save(fig, out / "curves.png")

    # ---- validation AUC per class
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(12, 2.6), sharey=True)
    for ax, c in zip(axes, CLASSES):
        for v in variants:
            band(ax, hist[hist.variant == v], f"val_auc_{c}", colors[v], label=labels[v])
        ax.set(title=c, xlabel="epoch")
    axes[0].set_ylabel("validation AUROC")
    # y-range from the mean curves after the first epochs, so the start-up ramp does
    # not flatten the part of the curve that matters
    late = hist[hist.epoch > min(5, hist.epoch.max() // 2)]
    lo = min(late.groupby(["variant", "epoch"])[f"val_auc_{c}"].mean().min() for c in CLASSES)
    axes[0].set_ylim(max(lo - 0.01, 0.5), None)
    handles = [Line2D([], [], color=colors[v], label=labels[v]) for v in variants]
    variant_legend(fig, handles, top=1.06)
    fig.suptitle(subtitle, x=0.01, ha="left", fontsize=8, color=MUTED, y=1.04)
    fig.tight_layout()
    save(fig, out / "val_auc_per_class.png")

    # ---- test AUC per class (dot plot: differences are ~0.01, bars from 0 would hide them)
    runs = collect_runs(exp_dir)
    if runs.empty:
        return
    rows = ["macro"] + CLASSES
    cols = ["test_macro_auc"] + [f"test_auc_{c}" for c in CLASSES]
    present = [v for v in variants if v in set(runs.variant)]
    n = len(present)
    fig, ax = plt.subplots(figsize=(5.6, len(rows) * (0.3 + 0.075 * n) + 0.9))
    spread = min(0.34, 0.09 * (n - 1))
    offsets = np.linspace(spread, -spread, n) if n > 1 else [0.0]  # first variant on top
    for off, v in zip(offsets, present):
        r = runs[runs.variant == v]
        mean, std = r[cols].mean().values, r[cols].std().fillna(0).values
        y = np.arange(len(rows))[::-1] + off
        ax.errorbar(mean, y, xerr=std, fmt="o", color=colors[v], markersize=7, elinewidth=1.5,
                    capsize=0, markeredgecolor="white", markeredgewidth=1.2,
                    label=f"{labels[v]} (n={len(r)})")
        ax.scatter(r[cols].values.ravel(), np.repeat(y, len(r)).reshape(len(rows), -1).T.ravel(),
                   s=8, color=colors[v], alpha=0.35, linewidths=0)
    ax.set_yticks(np.arange(len(rows))[::-1], rows, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("test AUROC (fold 10)   ·   dot = mean, line = ±1 std, faint = single seeds")
    ax.set_title("Test AUROC per class")
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.06), ncol=min(n, 3))
    save(fig, out / "test_auc_per_class.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", required=True)
    plot_experiment(p.parse_args().exp)
