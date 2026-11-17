"""Figures for the intermediate report and presentation (baseline B0), sized for slides.

    python -m src.visualize.intermediate_figures

Writes documentation/figures/intermediate/:
    b0_curves.png              validation macro AUROC and loss per epoch: resnet1d_wang (benchmark
                               procedure) and FCN (standard protocol), mean ± std over 3 seeds
    qualitative_examples.png   four test ECGs (lead II) with the predicted probability per class of
                               resnet1d_wang (best-validation checkpoint, seed 0) and the
                               per-class thresholds tuned on validation
    b0_subgroups.png           FCN test macro AUROC by age group and by sex, with 95% bootstrap CIs
                               (from runs/b0_baseline/subgroups/, src.evaluation.age_analysis)
    e1_val_auc.png             E1: validation macro AUROC per epoch of ResNet1D-10/-18/-34 (no augmentation)
    e3_val_auc.png             E3: validation macro AUROC per epoch of ResNet1D-10 for every augmentation
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.config import CLASSES, ROOT, RUNS_DIR
from src.data.dataset import load_splits
from src.visualize.plot_runs import INK, INK_2, MUTED, SERIES, band, load_histories, save

OUT = ROOT / "documentation" / "figures" / "intermediate"
WANG = RUNS_DIR / "b0_benchmark"
FCN = RUNS_DIR / "b0_baseline"
EXAMPLE_RUN = WANG / "resnet1d_wang_100hz_all12_aug-crop_std-global_benchmark_s0"
CLASS_COLORS = dict(zip(CLASSES, SERIES))
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})


def curves():
    models = [(load_histories(WANG), "resnet1d_wang (benchmark procedure)", SERIES[0]),
              (load_histories(FCN), "FCN (standard protocol)", SERIES[1])]
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.9), sharex=True)
    for hist, label, color in models:
        auc = band(axes[0], hist, "val_macro_auc", color, label=label)
        best = auc.idxmax()
        axes[0].plot(best, auc[best], "o", color=color, markersize=7, markeredgecolor="white",
                     markeredgewidth=1.5, zorder=5)
        band(axes[1], hist, "val_loss", color)
        band(axes[1], hist, "train_loss", color, linestyle="--", linewidth=1.4)
    axes[0].set(title="(a) Validation macro AUROC", ylim=(0.88, 0.94))
    axes[0].legend(fontsize=9.5, loc="lower right")
    axes[0].annotate("dot = best epoch", xy=(0.02, 0.92), xycoords="axes fraction", fontsize=9, color=MUTED)
    axes[1].set(title="(b) Loss (BCE)", xlabel="epoch", ylim=(0.1, 0.45))
    axes[1].legend(handles=[Line2D([], [], color=INK_2, label="validation"),
                            Line2D([], [], color=INK_2, linestyle="--", linewidth=1.4, label="train")],
                   fontsize=9.5, loc="upper right")
    fig.tight_layout()
    save(fig, OUT / "b0_curves.png")


def pick_examples(y, p, thr):
    """Four illustrative test records: a correct NORM, a correct multi-label prediction,
    a missed HYP and a false alarm on a normal ECG."""
    pred = p >= thr
    exact = (pred == y.astype(bool)).all(1)
    only = lambda *cs: (y[:, [CLASSES.index(c) for c in cs]].all(1) & (y.sum(1) == len(cs)))
    i_norm, i_mi, i_cd, i_hyp = (CLASSES.index(c) for c in ("NORM", "MI", "CD", "HYP"))
    a = np.flatnonzero(only("NORM") & exact)
    b = np.flatnonzero(only("MI", "CD") & exact)
    c = np.flatnonzero(y[:, i_hyp].astype(bool) & ~pred[:, i_hyp])
    d = np.flatnonzero(only("NORM") & pred[:, i_mi])
    return [
        ("Correct: normal ECG", a[np.argmax(p[a, i_norm])]),
        ("Correct multi-label: MI + CD", b[np.argmax(np.minimum(p[b, i_mi], p[b, i_cd]))]),
        ("Missed: hypertrophy", c[np.argmin(p[c, i_hyp])]),
        ("False alarm: MI on a normal ECG", d[np.argmax(p[d, i_mi])]),
    ]


def qualitative_examples():
    pred = np.load(EXAMPLE_RUN / "predictions.npz")
    thr = np.array([json.loads((EXAMPLE_RUN / "metrics.json").read_text())["thresholds_tuned"][c] for c in CLASSES])
    ids, y, p = pred["test_ecg_id"], pred["test_y"], pred["test_prob"]
    test = load_splits(100, "all12")["test"]
    assert (test.ecg_id == ids).all()
    lead2 = test.x[:, 1]  # raw mV, lead II

    examples = pick_examples(y, p, thr)
    fig, axes = plt.subplots(len(examples), 2, figsize=(5.3, 5.1), gridspec_kw={"width_ratios": [2.5, 1.3]})
    t = np.arange(lead2.shape[1]) / 100
    for row, (title, i) in enumerate(examples):
        true = [c for c, v in zip(CLASSES, y[i]) if v]
        predicted = [c for c, v in zip(CLASSES, p[i] >= thr) if v] or ["none"]
        ax = axes[row, 0]
        ax.plot(t, lead2[i], color=INK, linewidth=0.9)
        ax.set_xlim(0, 10)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, fontsize=9.5, loc="left")
        if row < len(examples) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time (s), lead II")
        bx = axes[row, 1]
        yy = np.arange(len(CLASSES))[::-1]
        bx.barh(yy, p[i], color=[CLASS_COLORS[c] for c in CLASSES], height=0.7)
        bx.scatter(thr, yy, marker="|", s=140, color=INK, linewidths=1.6, zorder=3)
        bx.set_xlim(0, 1)
        bx.set_title(f"true: {', '.join(true)}\npredicted: {', '.join(predicted)}", fontsize=8, loc="left",
                     color=INK_2, fontweight="normal")
        bx.set_yticks(yy, [f"{c} ●" if c in true else c for c in CLASSES], fontsize=8.5)
        for lbl, c in zip(bx.get_yticklabels(), CLASSES):
            lbl.set_fontweight("bold" if c in true else "normal")
        bx.tick_params(axis="x", labelsize=8.5)
        bx.grid(axis="y", visible=False)
        if row < len(examples) - 1:
            bx.set_xticklabels([])
        else:
            bx.set_xlabel("probability  (| = threshold)", fontsize=9)
    fig.tight_layout(h_pad=1.6)
    save(fig, OUT / "qualitative_examples.png")


def val_auc(exp, series, fname, legend_cols=1):
    """Validation macro AUROC per epoch (mean ± std over seeds) for (variant, label, colour, width)."""
    hist = load_histories(RUNS_DIR / exp)
    fig, ax = plt.subplots(figsize=(5.7, 1.95))
    lows = []
    for variant, label, color, width in series:
        h = hist[hist.variant == variant]
        auc = band(ax, h, "val_macro_auc", color, label=label, linewidth=width)
        best = auc.idxmax()
        ax.plot(best, auc[best], "o", color=color, markersize=5.5, markeredgecolor="white",
                markeredgewidth=1.2, zorder=5)
        lows.append(auc[auc.index > 5].min())
    ax.set_ylim(min(lows) - 0.004, None)
    ax.set_xlabel("epoch", fontsize=9.5)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=legend_cols,
              handlelength=1.6, borderaxespad=0)
    ax.annotate("dot = best epoch", xy=(0.99, 0.04), xycoords="axes fraction", ha="right", fontsize=8,
                color=MUTED)
    fig.tight_layout()
    save(fig, OUT / fname)


def e1_val_auc():
    val_auc("e1_depth", [(f"resnet1d{d}_100hz_all12", f"ResNet1D-{d}", SERIES[i], 1.8)
                         for i, d in enumerate((10, 18, 34))], "e1_val_auc.png")


def e3_val_auc():
    base = "resnet1d10_100hz_all12"
    series = [(base, "none", INK_2, 1.4),
              (base + "_aug-noise", "noise", SERIES[2], 1.4),
              (base + "_aug-wander", "baseline wander", SERIES[3], 1.4),
              (base + "_aug-scale", "scaling", SERIES[4], 1.4),
              (base + "_aug-shift", "time shift", SERIES[6], 1.4),
              (base + "_aug-crop", "random crop", SERIES[1], 2.4),
              (base + "_aug-noise+wander+scale+shift+crop", "all combined", SERIES[0], 2.4)]
    val_auc("e3_augment", series, "e3_val_auc.png")


def subgroups():
    sub = FCN / "subgroups"
    age = pd.read_csv(sub / "age_summary.csv").set_index("group")
    sex = pd.read_csv(sub / "sex_summary.csv").set_index("group")
    overall = age.loc["all", "macro_auc_mean"]
    panels = [(age.loc[["<40", "40-59", "60-79", "80+"]], "by age"), (sex.loc[["male", "female"]], "by sex")]
    fig, axes = plt.subplots(1, 2, figsize=(4.7, 1.85), sharey=True, gridspec_kw={"width_ratios": [2, 1.1]})
    for ax, (df, title) in zip(axes, panels):
        x = np.arange(len(df))
        lo, hi = df.macro_auc_ci_low_mean, df.macro_auc_ci_high_mean
        ax.errorbar(x, df.macro_auc_mean, yerr=[df.macro_auc_mean - lo, hi - df.macro_auc_mean], fmt="o",
                    color=SERIES[1], markersize=6, elinewidth=1.8, capsize=3)
        ax.axhline(overall, color=MUTED, linewidth=1, linestyle="--")
        ax.set_xticks(x, [f"{g.replace('-', '–')}\nn={int(n):,}" for g, n in zip(df.index, df.n_mean)], fontsize=8.5)
        ax.set_xlim(-0.5, len(df) - 0.5)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0.84, 0.96)
    axes[1].annotate(f"all: {overall:.3f}", xy=(0.98, 0.93), xycoords="axes fraction", ha="right", fontsize=8,
                     color=MUTED)
    fig.tight_layout(w_pad=0.6)
    save(fig, OUT / "b0_subgroups.png")


if __name__ == "__main__":
    curves()
    qualitative_examples()
    subgroups()
    e1_val_auc()
    e3_val_auc()
