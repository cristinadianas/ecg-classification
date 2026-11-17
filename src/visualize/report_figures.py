"""Figures for the intermediate report (LaTeX, NeurIPS layout, text width 5.5 in).

    python -m src.visualize.report_figures

Same style as the EDA and run figures (palette, fonts and helpers of plot_runs.py).
Writes documentation/figures/intermediate_report/:

    fig01_label_structure.png, fig03_example_ecg.png
                            copied unchanged from eda/figures/ (so documentation/ compiles on its own)
    age_distribution.png    unique patients per age group, and superclass prevalence per age group
    sampling_rates.png      lead I of one record at 100 Hz and 500 Hz (full record and one beat)
    b0_curves.png           B0 networks: validation macro AUROC and train/validation loss per epoch
    b0_test_metrics.png     B0: test AUROC and AUPRC (macro and per class) of every baseline
    b0_subgroups.png        FCN (B0): test macro AUROC by age and sex, per-class AUROC by age
    qualitative.png         four test ECGs with the predicted probabilities of resnet1d_wang
    experiments.png         E1 depth and E3 augmentation (validation curves), E4 leads (test AUROC)
"""

import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from matplotlib.lines import Line2D

from src.config import CLASSES, LEADS, RAW_DIR, ROOT, RUNS_DIR
from src.data.dataset import load_labels, load_splits
from src.evaluation.summarize import collect_runs
from src.visualize.intermediate_figures import EXAMPLE_RUN, pick_examples
from src.visualize.plot_runs import GRID, INK, INK_2, MUTED, SERIES, band, load_histories, save

OUT = ROOT / "documentation" / "figures" / "intermediate_report"
CLASS_COLORS = dict(zip(CLASSES, SERIES))
ACCENT = "#b23a4a"  # lead labels, as in the EDA figures
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})  # undo the slide sizes of intermediate_figures

AGE_GROUPS = ["0–9", "10–19", "20–29", "30–39", "40–49", "50–59", "60–69", "70–79", "80–89", "90+"]


def dot_legend(ax, **kw):
    ax.annotate("dot = best epoch", xycoords="axes fraction", fontsize=7.5, color=MUTED, **kw)


def end_labels(ax, x, ends, min_gap, fontsize=7.3):
    """Class names at the right end of line plots, pushed apart vertically so they never overlap.
    `ends` maps class -> y value of the last point."""
    order = sorted(ends, key=ends.get)
    ys = [ends[c] for c in order]
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + min_gap)
    shift = max(0.0, (ys[-1] - max(ends.values())) / 2)  # re-centre the stack around the data
    for c, y in zip(order, ys):
        ax.text(x, y - shift, c, va="center", fontsize=fontsize, color=CLASS_COLORS[c], fontweight="bold")


# ----------------------------------------------------------------------------- dataset

def metadata():
    """Records of PTB-XL v1.0.1 with age (placeholder 300 = 90+), sex and the 5 superclass labels."""
    db = pd.read_csv(RAW_DIR / "ptbxl_database.csv", index_col="ecg_id")
    labels = load_labels()
    df = db[["patient_id", "age", "sex", "filename_lr", "filename_hr"]].copy()
    for c in CLASSES:
        df[c] = labels[c].reindex(df.index).fillna(0).astype(bool)
    df["decade"] = pd.cut(df.age.clip(upper=90), bins=list(range(0, 100, 10)) + [400], right=False,
                          labels=AGE_GROUPS)
    return df


def fig_age(df):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) unique patients (age at the first record), as in the initial report
    patients = df.sort_index().groupby("patient_id").first()
    counts = patients.decade.value_counts().reindex(AGE_GROUPS)
    ax = axes[0]
    x = np.arange(len(AGE_GROUPS))
    ax.bar(x, counts.values, width=0.68, color=SERIES[0])
    for xi, v in zip(x, counts.values):
        ax.text(xi, v + 60, f"{v:,}", ha="center", fontsize=6.8, color=INK_2)
    ax.set_xticks(x, AGE_GROUPS, fontsize=7.3)
    ax.set_ylim(0, counts.max() * 1.13)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}K" if v else "0")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID)
    ax.set_xlabel(f"age (years)  ·  {len(patients):,} patients")
    ax.set_title("(a) Patients per age group")

    # (b) share of records carrying each superclass, per age group (records with a label)
    lab = df[df[CLASSES].any(axis=1) & df.age.notna()]
    groups = ["<40", "40–59", "60–79", "80+"]
    g = pd.cut(lab.age.clip(upper=90), [0, 40, 60, 80, 400], right=False, labels=groups)
    share = lab.groupby(g, observed=False)[CLASSES].mean()
    ax = axes[1]
    for c in CLASSES:
        ax.plot(range(len(groups)), share[c].values * 100, "o-", color=CLASS_COLORS[c], markersize=4.5,
                linewidth=1.8, markeredgecolor="white", markeredgewidth=0.8)
    end_labels(ax, len(groups) - 1 + 0.12, {c: share[c].values[-1] * 100 for c in CLASSES}, min_gap=6.5)
    n = g.value_counts().reindex(groups)
    ax.set_xticks(range(len(groups)), [f"{k}\nn={v:,}" for k, v in n.items()], fontsize=7.3)
    ax.set_xlim(-0.25, len(groups) - 0.45)
    ax.set_ylim(0, 85)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID)
    ax.set_title("(b) Records with each superclass, by age")
    fig.tight_layout(w_pad=2)
    save(fig, OUT / "age_distribution.png")


def fig_sampling(ecg_id=3, lead=0):
    """The same lead at 100 Hz and 500 Hz: full record and one heartbeat."""
    db = pd.read_csv(RAW_DIR / "ptbxl_database.csv", index_col="ecg_id")
    lo = wfdb.rdsamp(str(RAW_DIR / db.loc[ecg_id, "filename_lr"]))[0][:, lead]
    hi = wfdb.rdsamp(str(RAW_DIR / db.loc[ecg_id, "filename_hr"]))[0][:, lead]
    t_lo, t_hi = np.arange(len(lo)) / 100, np.arange(len(hi)) / 500
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 1.95), gridspec_kw={"width_ratios": [2.3, 1]})
    ax = axes[0]
    ax.plot(t_hi, hi, color=INK, linewidth=0.7)
    ax.set_xlim(0, 10)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mV")
    ax.set_title(f"(a) Lead I, 10 s at 500 Hz (record {ecg_id})")
    # one beat around the first R peak after 1 s
    r = 100 + int(np.argmax(lo[100:200]))
    a, b = (r - 25) / 100, (r + 45) / 100
    ax.axvspan(a, b, color=SERIES[1], alpha=0.12, linewidth=0)
    ax = axes[1]
    m_hi, m_lo = (t_hi >= a) & (t_hi <= b), (t_lo >= a) & (t_lo <= b)
    ax.plot(t_hi[m_hi], hi[m_hi], color=INK, linewidth=1.1, label="500 Hz (5,000 samples)")
    ax.plot(t_lo[m_lo], lo[m_lo], "o", color=SERIES[1], markersize=3.2, label="100 Hz (1,000 samples)")
    ax.set_xlim(a, b)
    ax.set_xlabel("time (s)")
    ax.set_title("(b) One heartbeat")
    ax.legend(fontsize=7, loc="upper right", handlelength=1.2, borderaxespad=0)
    fig.tight_layout(w_pad=1.5)
    save(fig, OUT / "sampling_rates.png")


# ----------------------------------------------------------------------------- B0

B0 = [  # (experiment, variant, label, colour)
    ("b0_benchmark", "resnet1d_wang_100hz_all12_aug-crop_std-global_benchmark", "resnet1d_wang", SERIES[0]),
    ("b0_baseline", "fcn1d_100hz_all12", "FCN", SERIES[1]),
    ("b0_svm", "svm-wavelet_100hz_all12", "SVM, ECG + wavelet features", SERIES[2]),
    ("b0_svm", "svm-statistical_100hz_all12", "SVM, statistical features", SERIES[3]),
]


def fig_b0_curves():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for exp, variant, label, color in B0[:2]:
        h = load_histories(RUNS_DIR / exp)
        auc = band(axes[0], h, "val_macro_auc", color, label=label + (" (benchmark procedure)" if
                                                                     "wang" in label else " (standard protocol)"))
        best = auc.idxmax()
        axes[0].plot(best, auc[best], "o", color=color, markersize=6, markeredgecolor="white",
                     markeredgewidth=1.2, zorder=5)
        band(axes[1], h, "val_loss", color)
        band(axes[1], h, "train_loss", color, linestyle="--", linewidth=1.3)
    axes[0].set(title="(a) Validation macro AUROC", xlabel="epoch", ylim=(0.88, 0.94))
    axes[0].legend(fontsize=7.5, loc="lower right")
    dot_legend(axes[0], xy=(0.02, 0.92))
    axes[1].set(title="(b) Loss (BCE)", xlabel="epoch", ylim=(0.1, 0.45))
    axes[1].legend(handles=[Line2D([], [], color=INK_2, label="validation"),
                            Line2D([], [], color=INK_2, linestyle="--", linewidth=1.3, label="train")],
                   fontsize=7.5, loc="upper right")
    fig.tight_layout(w_pad=2)
    save(fig, OUT / "b0_curves.png")


def fig_b0_test():
    """Dot plot of test AUROC and AUPRC (macro + per class); mean ± std over seeds, faint = single seeds."""
    runs = {label: collect_runs(RUNS_DIR / exp).query("variant == @variant") for exp, variant, label, _ in B0}
    rows = ["macro"] + CLASSES
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    offsets = np.linspace(0.27, -0.27, len(B0))
    for ax, metric, title in zip(axes, ["auc", "ap"], ["(a) Test AUROC", "(b) Test AUPRC"]):
        cols = [f"test_macro_{metric}"] + [f"test_{metric}_{c}" for c in CLASSES]
        for off, (_, _, label, color) in zip(offsets, B0):
            r = runs[label]
            y = np.arange(len(rows))[::-1] + off
            ax.errorbar(r[cols].mean().values, y, xerr=r[cols].std().fillna(0).values, fmt="o", color=color,
                        markersize=5, elinewidth=1.3, capsize=0, markeredgecolor="white", markeredgewidth=0.9,
                        label=label)
            if len(r) > 1:
                ax.scatter(r[cols].values.ravel(), np.tile(y, len(r)), s=6, color=color, alpha=0.35,
                           linewidths=0)
        for k in range(len(rows) - 1):
            ax.axhline(k + 0.5, color=GRID, linewidth=0.6)
        ax.set_yticks(np.arange(len(rows))[::-1], rows, color=INK)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
    axes[0].set_xlim(0.84, 0.96)
    axes[0].set_ylim(-0.62, 5.45)
    axes[0].axvline(0.930, color=MUTED, linestyle=":", linewidth=1)
    axes[0].text(0.9315, -0.42, "published 0.930", fontsize=6.5, color=MUTED, va="center")
    axes[1].set_xlim(0.6, 0.95)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.3, columnspacing=1.4)
    fig.text(0.5, -0.1, "dot = mean, line = ±1 std over 3 seeds, faint = single seeds (SVM: one deterministic run)",
             ha="center", fontsize=7, color=MUTED)
    fig.tight_layout(w_pad=1.5)
    save(fig, OUT / "b0_test_metrics.png")


def fig_b0_subgroups():
    sub = RUNS_DIR / "b0_baseline" / "subgroups"
    age = pd.read_csv(sub / "age_summary.csv").set_index("group")
    sex = pd.read_csv(sub / "sex_summary.csv").set_index("group")
    overall = age.loc["all", "macro_auc_mean"]
    groups = ["<40", "40-59", "60-79", "80+"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25), gridspec_kw={"width_ratios": [1.5, 0.85, 1.6]})
    for ax, df, title in [(axes[0], age.loc[groups], "(a) Macro AUROC by age"),
                          (axes[1], sex.loc[["male", "female"]], "(b) By sex")]:
        x = np.arange(len(df))
        lo, hi = df.macro_auc_ci_low_mean, df.macro_auc_ci_high_mean
        ax.errorbar(x, df.macro_auc_mean, yerr=[df.macro_auc_mean - lo, hi - df.macro_auc_mean], fmt="o",
                    color=SERIES[1], markersize=5, elinewidth=1.6, capsize=2.5)
        ax.axhline(overall, color=MUTED, linewidth=0.9, linestyle="--")
        ax.set_xticks(x, [f"{g.replace('-', '–')}\nn={int(n):,}" for g, n in zip(df.index, df.n_mean)],
                      fontsize=7.3)
        ax.set_xlim(-0.5, len(df) - 0.5)
        ax.set_ylim(0.84, 0.96)
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
    axes[1].set_yticklabels([])
    axes[0].text(3.45, overall + 0.002, f"all {overall:.3f}", ha="right", fontsize=7, color=MUTED)
    axes[0].set_ylabel("test macro AUROC (95% CI)")

    ax = axes[2]  # per-class AUROC by age; MI under 40 has 7 positives and is left out
    x = np.arange(len(groups))
    for c in CLASSES:
        v = age.loc[groups, f"auc_{c}_mean"].values.copy()
        pos = age.loc[groups, f"pos_{c}_mean"].values
        v[pos < 10] = np.nan
        ax.plot(x, v, "o-", color=CLASS_COLORS[c], markersize=4, linewidth=1.6, markeredgecolor="white",
                markeredgewidth=0.8)
    end_labels(ax, x[-1] + 0.12, {c: age.loc["80+", f"auc_{c}_mean"] for c in CLASSES}, min_gap=0.0125,
               fontsize=7)
    ax.set_xticks(x, [g.replace("-", "–") for g in groups], fontsize=7.3)
    ax.set_xlim(-0.25, len(groups) - 0.4)
    ax.set_ylim(0.8, 1.0)
    ax.grid(axis="x", visible=False)
    ax.set_title("(c) Per-class AUROC by age")
    fig.tight_layout(w_pad=1.2)
    save(fig, OUT / "b0_subgroups.png")


def fig_qualitative():
    """Four test records (lead II, raw mV) with the probabilities of resnet1d_wang (seed 0, best
    validation checkpoint) and the per-class thresholds tuned on validation."""
    pred = np.load(EXAMPLE_RUN / "predictions.npz")
    thr = np.array([json.loads((EXAMPLE_RUN / "metrics.json").read_text())["thresholds_tuned"][c] for c in CLASSES])
    ids, y, p = pred["test_ecg_id"], pred["test_y"], pred["test_prob"]
    test = load_splits(100, "all12")["test"]
    assert (test.ecg_id == ids).all()
    lead2 = test.x[:, 1]
    examples = pick_examples(y, p, thr)

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [2.2, 1, 2.2, 1]})
    t = np.arange(lead2.shape[1]) / 100
    for k, (title, i) in enumerate(examples):
        row, col = divmod(k, 2)
        ax, bx = axes[row, 2 * col], axes[row, 2 * col + 1]
        true = [c for c, v in zip(CLASSES, y[i]) if v]
        predicted = [c for c, v in zip(CLASSES, p[i] >= thr) if v] or ["none"]
        ax.plot(t, lead2[i], color=INK, linewidth=0.6)
        ax.set_xlim(0, 10)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.set_title(f"({'abcd'[k]}) {title}", fontsize=8)
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time (s), lead II", fontsize=7.5)
        yy = np.arange(len(CLASSES))[::-1]
        bx.barh(yy, p[i], color=[CLASS_COLORS[c] for c in CLASSES], height=0.7)
        bx.scatter(thr, yy, marker="|", s=60, color=INK, linewidths=1.3, zorder=3)
        bx.set_xlim(0, 1)
        bx.set_title(f"true: {', '.join(true)}\npred.: {', '.join(predicted)}", fontsize=6.8, loc="left",
                     color=INK_2, fontweight="normal")
        bx.set_yticks(yy, CLASSES, fontsize=6.8)
        for lbl, c in zip(bx.get_yticklabels(), CLASSES):
            lbl.set_fontweight("bold" if c in true else "normal")
            lbl.set_color(INK if c in true else MUTED)
        bx.tick_params(axis="x", labelsize=6.8)
        bx.tick_params(axis="y", length=0)
        bx.grid(axis="y", visible=False)
        if row == 0:
            bx.set_xticklabels([])
        else:
            bx.set_xlabel("probability", fontsize=7.5)
    fig.text(0.99, -0.03, "bold = true label  ·  | = per-class threshold tuned on validation",
             ha="right", fontsize=7, color=MUTED)
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    save(fig, OUT / "qualitative.png")


# ----------------------------------------------------------------------------- additional dataset / B0 figures

def fig_12leads(ecg_id=3):
    """All 12 leads of one record over the full 10 s: limb leads left, precordial leads right."""
    db = pd.read_csv(RAW_DIR / "ptbxl_database.csv", index_col="ecg_id")
    x = wfdb.rdsamp(str(RAW_DIR / db.loc[ecg_id, "filename_lr"]))[0]
    t = np.arange(len(x)) / 100
    lim = np.abs(x).max() * 1.05
    fig, axes = plt.subplots(6, 2, figsize=(7.2, 3.9), sharex=True, sharey=True)
    for col, (name, leads) in enumerate([("Limb leads", LEADS[:6]), ("Precordial leads", LEADS[6:])]):
        for row, lead in enumerate(leads):
            ax = axes[row, col]
            ax.plot(t, x[:, LEADS.index(lead)], color=INK, linewidth=0.7)
            ax.set_ylim(-lim, lim)
            ax.set_yticks([])
            ax.grid(axis="y", visible=False)
            ax.spines["left"].set_visible(False)
            ax.text(-0.02, 0.5, lead, transform=ax.transAxes, ha="right", va="center", fontsize=8,
                    fontweight="bold", color=ACCENT)
            if row < 5:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(axis="x", length=0)
        axes[0, col].set_title(f"({'ab'[col]}) {name}")
        axes[5, col].set_xlabel("time (s)")
    axes[5, 0].set_xlim(0, 10)
    fig.text(0.99, -0.01, f"record {ecg_id} (NORM), 100 Hz, same amplitude scale in every panel "
             f"(±{lim:.1f} mV)", ha="right", fontsize=7, color=MUTED)
    fig.tight_layout(h_pad=0.1, w_pad=2.5)
    save(fig, OUT / "ecg_12leads.png")


def fig_b0_roc_pr():
    """ROC and precision-recall curves per class of resnet1d_wang (seed 0) on the test fold."""
    from sklearn.metrics import auc, precision_recall_curve, roc_curve
    pred = np.load(EXAMPLE_RUN / "predictions.npz")
    y, p = pred["test_y"], pred["test_prob"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    for k, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y[:, k], p[:, k])
        axes[0].plot(fpr, tpr, color=CLASS_COLORS[c], linewidth=1.5, label=f"{c} ({auc(fpr, tpr):.3f})")
        prec, rec, _ = precision_recall_curve(y[:, k], p[:, k])
        axes[1].plot(rec, prec, color=CLASS_COLORS[c], linewidth=1.5, label=f"{c} ({auc(rec, prec):.3f})")
        axes[1].axhline(y[:, k].mean(), color=CLASS_COLORS[c], linewidth=0.8, linestyle=":")
    axes[0].plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle="--")
    axes[0].set(title="(a) ROC curves", xlabel="false-positive rate", ylabel="true-positive rate",
                xlim=(0, 1), ylim=(0, 1.01))
    axes[1].set(title="(b) Precision-recall curves", xlabel="recall", ylabel="precision", xlim=(0, 1),
                ylim=(0, 1.01))
    axes[0].legend(fontsize=7, loc="lower right", title="class (AUROC)", title_fontsize=7)
    axes[1].legend(fontsize=7, loc="lower left", title="class (AUPRC)", title_fontsize=7)
    axes[1].annotate("dotted = prevalence (chance level)", xy=(0.98, 0.02), xycoords="axes fraction",
                     ha="right", fontsize=6.8, color=MUTED)
    for ax in axes:
        ax.set_aspect("equal")
    fig.tight_layout(w_pad=2.5)
    save(fig, OUT / "b0_roc_pr.png")


# ----------------------------------------------------------------------------- experiments E1-E4

def curves_panel(ax, exp, series, col="val_macro_auc", best=True, legend=True, ncol=3):
    """Mean ± std curves of `col` for (variant, label, colour, width[, linestyle])."""
    hist = load_histories(RUNS_DIR / exp)
    for variant, label, color, width, *style in series:
        kw = dict(linestyle=style[0]) if style else {}
        m = band(ax, hist[hist.variant == variant], col, color, label=label, linewidth=width, **kw)
        if best:
            b = m.idxmax()
            ax.plot(b, m[b], "o", color=color, markersize=4.5, markeredgecolor="white", markeredgewidth=0.9,
                    zorder=5)
    ax.set_xlabel("epoch")
    if legend:
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=ncol, handlelength=1.4,
                  columnspacing=1.0)


def class_dots(ax, groups, xlim):
    """Test AUROC (macro + per class): mean ± std over seeds for each (runs, label, colour)."""
    rows = ["macro"] + CLASSES
    cols = ["test_macro_auc"] + [f"test_auc_{c}" for c in CLASSES]
    offsets = np.linspace(0.24, -0.24, len(groups))
    for off, (r, label, color) in zip(offsets, groups):
        y = np.arange(len(rows))[::-1] + off
        ax.errorbar(r[cols].mean().values, y, xerr=r[cols].std().values, fmt="o", color=color, markersize=4.5,
                    elinewidth=1.2, markeredgecolor="white", markeredgewidth=0.8, label=label)
    for k in range(len(rows) - 1):
        ax.axhline(k + 0.5, color=GRID, linewidth=0.6)
    ax.set_yticks(np.arange(len(rows))[::-1], rows, color=INK, fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(*xlim)
    ax.set_xlabel("test AUROC (mean ± std over 3 seeds)")


def bottom_legend(fig, labels_colors, marker=None):
    fig.legend(handles=[Line2D([], [], color=c, marker=marker, label=l) for l, c in labels_colors],
               loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=len(labels_colors), fontsize=7.5)


def loss_panel(ax, exp, series, log=False):
    curves_panel(ax, exp, series, col="val_loss", best=False, legend=False)
    curves_panel(ax, exp, [(v, None, c, 1.2, "--") for v, _, c, _ in series], col="train_loss", best=False,
                 legend=False)
    if log:
        ax.set_yscale("log")
    ax.legend(handles=[Line2D([], [], color=INK_2, label="validation"),
                       Line2D([], [], color=INK_2, linestyle="--", linewidth=1.2, label="train")],
              fontsize=7.5, loc="lower left" if log else "upper right")


def fig_e1():
    series = [(f"resnet1d{d}_100hz_all12", f"ResNet1D-{d}", SERIES[i], 1.5) for i, d in enumerate((10, 18, 34))]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    curves_panel(axes[0], "e1_depth", series, legend=False)
    axes[0].set(title="(a) Validation macro AUROC", ylim=(0.88, 0.925))
    dot_legend(axes[0], xy=(0.98, 0.04), ha="right")
    loss_panel(axes[1], "e1_depth", series, log=True)
    axes[1].set_title("(b) Loss (BCE, log scale)")
    bottom_legend(fig, [(l, c) for _, l, c, _ in series])
    fig.tight_layout(w_pad=2)
    save(fig, OUT / "e1_depth.png")


E2_VARIANTS = [("resnet1d10_100hz_all12_aug-crop", "convolution (ResNet1D-10)", SERIES[0]),
               ("bilstm_100hz_all12_aug-crop_fp16lstm", "recurrence (BiLSTM)", SERIES[1]),
               ("cnn_transformer_100hz_all12_aug-crop", "self-attention (CNN + Transformer)", SERIES[2])]


def fig_e2():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), gridspec_kw={"width_ratios": [1.1, 1]})
    curves_panel(axes[0], "e2_operation_simple", [(v, l, c, 1.5) for v, l, c in E2_VARIANTS], legend=False)
    axes[0].set(title="(a) Validation macro AUROC", ylim=(0.905, 0.935))
    dot_legend(axes[0], xy=(0.98, 0.04), ha="right")
    runs = collect_runs(RUNS_DIR / "e2_operation_simple")
    class_dots(axes[1], [(runs[runs.variant == v], l, c) for v, l, c in E2_VARIANTS], (0.89, 0.955))
    axes[1].set_title("(b) Test AUROC per class")
    bottom_legend(fig, [(l, c) for _, l, c in E2_VARIANTS])
    fig.tight_layout(w_pad=2)
    save(fig, OUT / "e2_operation.png")


E3_CONDITIONS = [("", "none", INK_2), ("_aug-noise", "noise", SERIES[2]),
                 ("_aug-wander", "baseline wander", SERIES[3]), ("_aug-scale", "scaling", SERIES[4]),
                 ("_aug-shift", "time shift", SERIES[6]), ("_aug-crop", "random crop", SERIES[1]),
                 ("_aug-noise+wander+scale+shift+crop", "all combined", SERIES[0])]


def fig_e3():
    base = "resnet1d10_100hz_all12"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), gridspec_kw={"width_ratios": [1.25, 1]})
    curves_panel(axes[0], "e3_augment", [(base + s, l, c, 1.8 if s in ("_aug-crop", "_aug-shift") else 1.1)
                                         for s, l, c in E3_CONDITIONS[:-1]], ncol=3)
    axes[0].set(title="(a) Validation macro AUROC", ylim=(0.885, 0.935))
    dot_legend(axes[0], xy=(0.98, 0.04), ha="right")

    ax = axes[1]  # gain in test macro AUROC over no augmentation, per seed
    runs = collect_runs(RUNS_DIR / "e3_augment")
    ref = runs[runs.variant == base].test_macro_auc.mean()
    conds = E3_CONDITIONS[1:]
    for k, (s, label, color) in enumerate(conds):
        d = runs[runs.variant == base + s].test_macro_auc.values - ref
        yk = len(conds) - 1 - k
        ax.barh(yk, d.mean(), height=0.6, color=color, alpha=0.85)
        ax.scatter(d, [yk] * len(d), s=9, color=INK, zorder=3, linewidths=0)
        # label = difference of the rounded means, the convention of the README / report tables
        label_delta = round(d.mean() + ref, 4) - round(ref, 4)
        ax.text(max(d.max(), 0) + 0.0006, yk, f"{label_delta:+.4f}", va="center", fontsize=7, color=INK_2)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(range(len(conds))[::-1], [l for _, l, _ in conds], fontsize=7.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.003, 0.0215)
    ax.set_xlabel("Δ test macro AUROC vs. none (dots = seeds)")
    ax.set_title("(b) Gain over no augmentation")
    fig.tight_layout(w_pad=2)
    save(fig, OUT / "e3_augment.png")


def fig_e4():
    sets = [("all12", "12 leads", SERIES[0]), ("limb6", "6 limb leads", SERIES[1]), ("lead2", "lead II", SERIES[2])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), gridspec_kw={"width_ratios": [1, 1.1]})
    runs = collect_runs(RUNS_DIR / "e4_leads")
    class_dots(axes[0], [(runs[runs.variant == f"resnet1d10_100hz_{t}_aug-crop"], l, c) for t, l, c in sets],
               (0.79, 0.96))
    axes[0].set_title("(a) Test AUROC per class")
    loss_panel(axes[1], "e4_leads", [(f"resnet1d10_100hz_{t}_aug-crop", l, c, 1.5) for t, l, c in sets])
    axes[1].set(title="(b) Loss (BCE)", ylim=(0.1, 0.5))
    bottom_legend(fig, [(l, c) for _, l, c in sets])
    fig.tight_layout(w_pad=2)
    save(fig, OUT / "e4_leads.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "eda" / "figures" / "fig01_label_structure.png", OUT / "fig01_label_structure.png")
    df = metadata()
    fig_age(df)
    fig_sampling()
    fig_12leads()
    fig_b0_curves()
    fig_b0_test()
    fig_b0_roc_pr()
    fig_b0_subgroups()
    fig_qualitative()
    fig_e1()
    fig_e2()
    fig_e3()
    fig_e4()
