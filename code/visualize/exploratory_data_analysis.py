"""Exploratory data analysis of PTB-XL for the 5-superclass multi-label task.

Run from the project root:
    python src/visualize/exploratory_data_analysis.py

Writes figures to eda/figures/ (report versions) and eda/figures/slides/
(versions sized for the presentation), and the numbers quoted in the report to
eda/eda_summary.json.
"""

import ast
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from scipy.signal import butter, filtfilt, find_peaks

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "ptb-xl"
OUT = ROOT / "eda" / "figures"
OUT_SLIDES = OUT / "slides"

FS = 100  # we model the 100 Hz version
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_NAMES = {
    "NORM": "Normal",
    "MI": "Myocardial infarction",
    "STTC": "ST/T change",
    "CD": "Conduction disturbance",
    "HYP": "Hypertrophy",
}
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
QUALITY_FLAGS = {
    "static_noise": "Static noise",
    "extra_beats": "Extra beats",
    "baseline_drift": "Baseline drift",
    "burst_noise": "Burst noise",
    "pacemaker": "Pacemaker",
    "electrodes_problems": "Electrode problems",
}

# Colours: one fixed hue per class, used in every figure. Ink matches the slide deck.
CLASS_COLORS = dict(zip(CLASSES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
INK = "#14213d"
INK_2 = "#4a5568"
MUTED = "#898781"
GRID = "#e1e0d9"
ACCENT = "#b23a4a"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#86b6ef", "#f0efec", "#ef9a8c", "#b4302f"])

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
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "text.color": INK,
    "legend.frameon": False,
    "savefig.dpi": 220,
    "savefig.transparent": True,
})


# ----------------------------------------------------------------------------- data

def load_metadata():
    df = pd.read_csv(DATA / "ptbxl_database.csv", index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    scp = pd.read_csv(DATA / "scp_statements.csv", index_col=0)
    diag = scp[scp.diagnostic == 1]
    # Same aggregation as the PhysioNet example and the Strodthoff et al. benchmark:
    # every diagnostic statement maps to its superclass, whatever its likelihood.
    df["superclasses"] = df.scp_codes.apply(
        lambda codes: sorted({diag.loc[c, "diagnostic_class"] for c in codes if c in diag.index},
                             key=CLASSES.index))
    df["n_labels"] = df.superclasses.apply(len)
    for c in CLASSES:
        df[c] = df.superclasses.apply(lambda s, c=c: c in s)
    df["any_quality_flag"] = df[list(QUALITY_FLAGS)].notna().any(axis=1)
    return df, scp


def load_signals(df):
    """All records at 100 Hz as float32, shape (n_records, 1000, 12), in mV."""
    return np.stack([wfdb.rdsamp(str(DATA / f))[0].astype(np.float32) for f in df.filename_lr])


def highpass(x, cutoff=0.5):
    b, a = butter(2, cutoff / (FS / 2), btype="highpass")
    return filtfilt(b, a, x, axis=0)


def beat_template(x, pre=0.25, post=0.45):
    """Median beat of one record (all 12 leads), aligned on the R peak.

    R peaks: maxima of the smoothed, lead-summed squared derivative, refined to the
    largest absolute amplitude summed over leads. The template is referenced to the
    PR segment (100-60 ms before R), which is how ST deviation is read clinically.
    """
    x = highpass(x)
    energy = np.convolve((np.diff(x, axis=0) ** 2).sum(axis=1), np.ones(5) / 5, mode="same")
    peaks, _ = find_peaks(energy, distance=int(0.3 * FS), height=0.3 * np.percentile(energy, 99))
    a, b = int(pre * FS), int(post * FS)
    amp = np.abs(x).sum(axis=1)
    beats = []
    for p in peaks:
        lo, hi = max(p - 6, 0), min(p + 6, len(x))
        r = lo + int(np.argmax(amp[lo:hi]))
        if r - a >= 0 and r + b <= len(x):
            beats.append(x[r - a:r + b])
    if len(beats) < 3:
        return None
    tpl = np.median(beats, axis=0)
    return tpl - tpl[a - 10:a - 6].mean(axis=0)


# ----------------------------------------------------------------------------- helpers

def save(fig, name, folder=OUT):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", (folder / name).relative_to(ROOT))


def hgrid(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def vgrid(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def alone_vs_shared(df):
    alone = [int(((df[c]) & (df.n_labels == 1)).sum()) for c in CLASSES]
    total = [int(df[c].sum()) for c in CLASSES]
    return alone, total


def stacked_class_bars(ax, df, label_size=9, legend=True):
    """Records per superclass, split into 'only this label' and 'with other labels'."""
    alone, total = alone_vs_shared(df)
    y = np.arange(len(CLASSES))[::-1]
    for yi, c, a, t in zip(y, CLASSES, alone, total):
        ax.barh(yi, a, height=0.62, color=CLASS_COLORS[c], edgecolor="none")
        ax.barh(yi, t - a, left=a + 40, height=0.62, color=CLASS_COLORS[c], alpha=0.32, edgecolor="none")
        ax.text(t + 120, yi, f"{t:,}  ({t / len(df):.0%})", va="center", fontsize=label_size, color=INK_2)
    ax.set_yticks(y, CLASSES, fontsize=label_size + 0.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(total) * 1.32)
    ax.xaxis.set_major_locator(MultipleLocator(2000 if label_size < 10 else 4000))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}K" if v else "0")
    vgrid(ax)
    ax.spines["left"].set_visible(False)
    if legend:
        ax.legend(handles=[Patch(color="#6b6a66", label="only this label"),
                           Patch(color="#6b6a66", alpha=0.32, label="with other labels")],
                  loc="lower right", fontsize=label_size - 0.5, handlelength=1.2)


# ----------------------------------------------------------------------------- figures

def fig_labels(df, summary):
    """Imbalance, label cardinality and co-occurrence in one row."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw={"width_ratios": [1.45, 0.8, 1.05]})

    ax = axes[0]
    stacked_class_bars(ax, df, label_size=8)
    ax.set_title("(a) Records per superclass")
    ax.set_xlabel("records")

    ax = axes[1]
    counts = df.n_labels.value_counts().sort_index()
    ax.bar(counts.index, counts.values, width=0.62, color="#3987e5")
    for k, v in counts.items():
        ax.text(k, v + 300, f"{v / len(df):.0%}" if v / len(df) >= 0.01 else f"{v / len(df):.1%}",
                ha="center", fontsize=7.5, color=INK_2)
    ax.set_xticks(counts.index)
    ax.set_xlabel("superclasses per record")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}K" if v else "0")
    ax.set_ylim(0, counts.max() * 1.12)
    hgrid(ax)
    ax.set_title("(b) Labels per record")

    ax = axes[2]
    m = df[CLASSES].to_numpy(dtype=float)
    cond = (m.T @ m) / m.sum(axis=0)[:, None]  # row A, column B: P(B | A)
    off_diag = np.where(np.eye(5, dtype=bool), np.nan, cond)
    ax.imshow(off_diag, cmap=SEQ, vmin=0, vmax=0.6)
    for i in range(5):
        for j in range(5):
            v = cond[i, j]
            if i == j:
                ax.text(j, i, "-", ha="center", va="center", fontsize=7.3, color=MUTED)
            else:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7.3,
                        color="white" if v > 0.33 else INK)
    ax.set_xticks(range(5), CLASSES, fontsize=7.3, rotation=0)
    ax.set_yticks(range(5), CLASSES, fontsize=7.3)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("... also has this label", fontsize=8)
    ax.set_ylabel("record with this label ...", fontsize=8)
    ax.set_title("(c) Co-occurrence P(col | row)")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig01_label_structure.png")

    summary["conditional_cooccurrence"] = {
        a: {b: round(float(cond[i, j]), 3) for j, b in enumerate(CLASSES)} for i, a in enumerate(CLASSES)}


def fig_label_combinations(df):
    """The 12 most frequent exact label sets (an UpSet-style view)."""
    combos = Counter(tuple(s) for s in df.superclasses).most_common(12)
    fig, (top, mat) = plt.subplots(2, 1, figsize=(7.2, 3.1), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.05})
    x = np.arange(len(combos))
    vals = [n for _, n in combos]
    top.bar(x, vals, width=0.62, color="#3987e5")
    for xi, v in zip(x, vals):
        top.text(xi, v * 1.15, f"{v:,}", ha="center", fontsize=7.5, color=INK_2)
    top.set_yscale("log")
    top.set_ylim(80, 40000)
    top.set_ylabel("records (log scale)")
    hgrid(top)
    top.set_title("Most frequent exact label combinations")
    for yi, c in enumerate(CLASSES):
        mat.scatter(x, [yi] * len(x), s=22, color="#e1e0d9", zorder=1)
        on = [xi for xi, (combo, _) in zip(x, combos) if c in combo]
        mat.scatter(on, [yi] * len(on), s=30, color=CLASS_COLORS[c], zorder=2,
                    edgecolors="white", linewidths=1)
    for xi, (combo, _) in zip(x, combos):
        ys = [CLASSES.index(c) for c in combo]
        if len(ys) > 1:
            mat.plot([xi, xi], [min(ys), max(ys)], color=MUTED, linewidth=1.5, zorder=0)
    mat.set_yticks(range(5), CLASSES, fontsize=8, color=INK)
    mat.invert_yaxis()
    mat.set_xticks(x, ["none" if not c else "" for c, _ in combos], fontsize=7.5)
    mat.tick_params(length=0)
    for s in mat.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    save(fig, "fig02_label_combinations.png")


def draw_superclass_matrix(ax, both, alone, fs=8):
    """Off-diagonal: records with both labels. Diagonal (grey): records with only that label."""
    off = np.where(np.eye(5, dtype=bool), np.nan, both)
    ax.imshow(off, cmap=SEQ, vmin=0, vmax=np.nanmax(off))
    for i in range(5):
        for j in range(5):
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#f0efec"))
                ax.text(j, i - 0.12, f"{alone[i]:,}", ha="center", va="center", fontsize=fs, color=INK_2)
                ax.text(j, i + 0.24, "alone", ha="center", va="center", fontsize=fs * 0.8, color=MUTED)
            else:
                v = both[i, j]
                ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=fs,
                        color="white" if v > 0.5 * np.nanmax(off) else INK)
    ax.set_xticks(range(5), CLASSES, fontsize=fs)
    ax.set_yticks(range(5), CLASSES, fontsize=fs)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig_cooccurrence(df, scp, summary):
    """Which diagnoses overlap: superclass and subclass co-occurrence counts, and the top pairs."""
    # (1) Superclasses, confusion-matrix style. Off-diagonal: records with both labels.
    # Diagonal: records with only that label, drawn in grey so it does not dominate the colour scale.
    m = df[CLASSES].to_numpy(dtype=int)
    both = m.T @ m
    alone = [int((df[c] & (df.n_labels == 1)).sum()) for c in CLASSES]
    fig, ax = plt.subplots(figsize=(3.9, 3.4))
    draw_superclass_matrix(ax, both, alone)
    ax.set_title("Records carrying both superclasses")
    fig.tight_layout()
    save(fig, "fig10_cooccurrence_superclass.png")

    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    draw_superclass_matrix(ax, both, alone, fs=11)
    fig.tight_layout()
    save(fig, "slide_cooccurrence.png", OUT_SLIDES)

    # (2) Diagnostic subclasses, grouped by superclass, lower triangle (the matrix is symmetric).
    diag = scp[scp.diagnostic == 1]
    subs_of = df.scp_codes.apply(
        lambda codes: {diag.loc[c, "diagnostic_subclass"] for c in codes if c in diag.index})
    totals = Counter(x for s in subs_of for x in s)
    def nice(g):
        return "AVB" if g == "_AVB" else g

    order = []
    for c in CLASSES:
        group = diag.loc[diag.diagnostic_class == c, "diagnostic_subclass"].unique()
        order += sorted(group, key=lambda g: -totals[g])
    k = len(order)
    pos = {g: i for i, g in enumerate(order)}
    cnt = np.zeros((k, k), dtype=int)
    for s in subs_of:
        idx = [pos[g] for g in s]
        for a in idx:
            for b in idx:
                cnt[a, b] += 1
    lower = np.where(np.tril(np.ones((k, k), dtype=bool), -1), cnt, np.nan)
    lower[lower == 0] = np.nan
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    im = ax.imshow(lower, cmap=SEQ, norm=PowerNorm(0.45, vmin=1, vmax=np.nanmax(lower)))
    for i in range(k):
        ax.text(i, i, f"{cnt[i, i]:,}", ha="center", va="center", fontsize=5.6, color=MUTED, rotation=45)
        for j in range(i):
            v = cnt[i, j]
            if v >= 50:
                ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=5.6,
                        color="white" if v > 300 else INK)
    labels = [nice(g) for g in order]
    ax.set_xticks(range(k), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(k), labels, fontsize=7)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    start = 0
    for c in CLASSES:
        n = int((diag.diagnostic_class == c).groupby(diag.diagnostic_subclass).any().sum())
        if start:
            ax.plot([-0.5, start - 0.5], [start - 0.5, start - 0.5], color=INK_2, linewidth=0.8)
            ax.plot([start - 0.5, start - 0.5], [start - 0.5, k - 0.5], color=INK_2, linewidth=0.8)
        ax.text(k + 0.2, start + (n - 1) / 2, c, va="center", fontsize=8.5, fontweight="bold",
                color=CLASS_COLORS[c] if c != "CD" else "#b07800")
        start += n
    ax.set_xlim(-0.5, k + 1.6)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[10, 100, 300, 600, 1000])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=0)
    cb.set_label("records with both labels", fontsize=8, color=INK_2)
    ax.set_title("Co-occurrence of diagnostic subclasses (diagonal: records with the subclass; "
                 "lines separate superclasses)", fontsize=9)
    fig.tight_layout()
    save(fig, "fig11_cooccurrence_subclass.png")

    # (3) The 15 most frequent subclass pairs. Bar label: share of the rarer label that the pair covers.
    sup_of = dict(zip(diag.diagnostic_subclass, diag.diagnostic_class))
    pairs = sorted(((cnt[i, j], order[j], order[i]) for i in range(k) for j in range(i)), reverse=True)[:15]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(pairs))[::-1]
    for yi, (v, a, b) in zip(y, pairs):
        across = sup_of[a] != sup_of[b]
        ax.barh(yi, v, height=0.62, color="#eb6834" if across else "#2a78d6")
        rare = a if totals[a] <= totals[b] else b
        ax.text(v + 12, yi, f"{v:,}  ·  {v / totals[rare]:.0%} of {nice(rare)}", va="center", fontsize=7.5,
                color=INK_2)
    ax.set_yticks(y, [f"{nice(a)} + {nice(b)}  ({sup_of[a]}{'' if sup_of[a] == sup_of[b] else ' + ' + sup_of[b]})"
                      for _, a, b in pairs], fontsize=7.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, pairs[0][0] * 1.32)
    vgrid(ax)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("records with both labels")
    ax.legend(handles=[Patch(color="#eb6834", label="across superclasses (makes the task multi-label)"),
                       Patch(color="#2a78d6", label="within one superclass")],
              loc="lower right", fontsize=7.5, handlelength=1.2)
    ax.set_title("Most frequent overlapping diagnostic subclasses")
    fig.tight_layout()
    save(fig, "fig12_top_overlapping_pairs.png")

    summary["superclass_cooccurrence_counts"] = {
        a: {b: int(both[i, j]) for j, b in enumerate(CLASSES)} for i, a in enumerate(CLASSES)}
    summary["top_subclass_pairs"] = [
        {"pair": [a, b], "records": int(v), "superclasses": [sup_of[a], sup_of[b]],
         "share_of_rarer": round(v / min(totals[a], totals[b]), 3)} for v, a, b in pairs]


def pick_example(df, cls):
    """A clean, human-validated record whose only superclass is `cls`, with 100% likelihood."""
    ok = (df.n_labels == 1) & df[cls] & ~df.any_quality_flag & df.validated_by_human & (df.strat_fold <= 8)
    ok &= df.scp_codes.apply(lambda c: max(c.values()) == 100)
    return df.index[ok][0]


def draw_12lead(ax, x, t, lw=0.9, label_fs=8):
    """Clinical 3x4 layout (2.5 s per lead) with a full lead II rhythm strip underneath."""
    layout = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
    off = 3.0
    for r, row in enumerate(layout + [["II"]]):
        base = -r * off
        for k, lead in enumerate(row):
            seg = slice(k * 250, (k + 1) * 250) if len(row) == 4 else slice(0, 1000)
            ax.plot(t[seg], x[seg, LEADS.index(lead)] + base, color=INK, linewidth=lw)
            ax.text(t[seg][0] + 0.05, base + 1.0, lead, fontsize=label_fs, color=ACCENT, fontweight="bold")
    ax.set_xlim(0, 10)
    ax.set_xticks(range(11))
    ax.set_xlabel("time (s)  ·  100 Hz, 1000 samples per lead")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)


def fig_example_ecg(df, X, summary):
    """Clinical 3x4 layout plus a lead II rhythm strip, one NORM record."""
    ecg_id = pick_example(df, "NORM")
    x = X[df.index.get_loc(ecg_id)]
    x = highpass(x)
    t = np.arange(len(x)) / FS
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    draw_12lead(ax, x, t)
    ax.set_title(f"Record {ecg_id} (NORM): 12 leads, 10 s; bottom row is lead II over the full record")
    fig.tight_layout()
    save(fig, "fig03_example_ecg.png")
    summary["example_ecg_id"] = int(ecg_id)

    # Slide version: lead II rhythm strip, wide and short.
    fig, ax = plt.subplots(figsize=(11.6, 1.05))
    ax.plot(t, x[:, 1], color=INK, linewidth=1.1)
    ax.set_xlim(0, 10)
    ax.axis("off")
    save(fig, "slide_lead2_strip.png", OUT_SLIDES)

    # Title slide: same strip in light ink for a dark background.
    fig, ax = plt.subplots(figsize=(11.6, 1.05))
    ax.plot(t, x[:, 1], color="#f7f5f0", linewidth=1.4, alpha=0.55)
    ax.set_xlim(0, 10)
    ax.axis("off")
    save(fig, "slide_title_strip.png", OUT_SLIDES)

    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    draw_12lead(ax, x, t, lw=1.0, label_fs=11)
    ax.tick_params(labelsize=10)
    ax.set_xlabel("time (s)  ·  100 Hz  ·  1000 samples × 12 leads", fontsize=10.5)
    fig.tight_layout()
    save(fig, "slide_example_12lead.png", OUT_SLIDES)


def fig_median_beats(df, X, summary, per_class=600, seed=0):
    """Median heartbeat per superclass in three leads, from clean single-label records."""
    rng = np.random.default_rng(seed)
    leads = ["II", "V2", "V5"]
    templates, used = {}, {}
    for c in CLASSES:
        pool = df.index[(df.n_labels == 1) & df[c] & ~df.any_quality_flag & (df.pacemaker.isna())]
        pick = rng.choice(pool, size=min(per_class, len(pool)), replace=False)
        tpls = [beat_template(X[df.index.get_loc(i)]) for i in pick]
        tpls = np.array([tp for tp in tpls if tp is not None])
        templates[c], used[c] = tpls, len(tpls)
    t = (np.arange(templates["NORM"].shape[1]) - int(0.25 * FS)) / FS * 1000

    def draw(axes, lw=1.6, fs=8, leads=leads):
        for ax, lead in zip(axes, leads):
            li = LEADS.index(lead)
            for c in CLASSES:
                med = np.median(templates[c][:, :, li], axis=0)
                ax.plot(t, med, color=CLASS_COLORS[c], linewidth=lw, label=c, solid_capstyle="round")
            ax.axhline(0, color="#c3c2b7", linewidth=0.8, zorder=0)
            ax.set_title(f"Lead {lead}", fontsize=fs + 1.5)
            ax.set_xlim(t[0], t[-1])
            ax.set_xticks([-200, 0, 200, 400])
            ax.tick_params(labelsize=fs)
            hgrid(ax)
        axes[0].set_ylabel("mV (vs. PR segment)", fontsize=fs)
        for ax in axes:
            ax.set_xlabel("ms from R peak", fontsize=fs)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharey=False)
    draw(axes)
    axes[-1].legend(loc="upper right", fontsize=7.5, handlelength=1.2, labelspacing=0.25)
    fig.suptitle("Median heartbeat per superclass (single-label, artefact-free records)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig04_median_beats.png")

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.6))
    draw(axes, lw=2.2, fs=13, leads=["II", "V5"])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=13,
               bbox_to_anchor=(0.5, -0.13), handlelength=1.2, columnspacing=1.2)
    fig.tight_layout(w_pad=1.0)
    save(fig, "slide_median_beats.png", OUT_SLIDES)

    # Quantify what the plot shows: ST level (80 ms after R relative to PR), R amplitude, QRS width proxy.
    r0 = int(0.25 * FS)
    stats = {}
    for c in CLASSES:
        tp = templates[c]
        stats[c] = {
            "records_used": used[c],
            "st_level_V2_mV": round(float(np.median(tp[:, r0 + 8, LEADS.index("V2")])), 3),
            "st_level_V5_mV": round(float(np.median(tp[:, r0 + 8, LEADS.index("V5")])), 3),
            "r_amp_V5_mV": round(float(np.median(tp[:, r0, LEADS.index("V5")])), 3),
            "s_depth_V2_mV": round(float(np.median(tp[:, r0:r0 + 8, LEADS.index("V2")].min(axis=1))), 3),
        }
    summary["median_beats"] = stats


def fig_demographics(df, summary):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    data = [df.loc[df[c], "age"].dropna().to_numpy() for c in CLASSES]
    y = np.arange(5)[::-1]
    bp = ax.boxplot(data, positions=y, vert=False, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "linewidth": 1.6},
                    whiskerprops={"color": MUTED}, capprops={"color": MUTED})
    for patch, c in zip(bp["boxes"], CLASSES):
        patch.set_facecolor(CLASS_COLORS[c])
        patch.set_alpha(0.75)
        patch.set_edgecolor("none")
    for yi, d in zip(y, data):
        ax.text(101, yi, f"median {np.median(d):.0f}", va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(y, CLASSES, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("age (years; >89 capped)")
    vgrid(ax)
    ax.spines["left"].set_visible(False)
    ax.set_title("(a) Age by superclass")

    ax = axes[1]
    male = [float((df.loc[df[c], "sex"] == 0).mean()) for c in CLASSES]
    ax.barh(y, male, height=0.55, color=[CLASS_COLORS[c] for c in CLASSES])
    overall = (df.sex == 0).mean()
    ax.axvline(overall, color=INK_2, linewidth=1, linestyle=(0, (3, 2)))
    ax.text(overall + 0.015, -0.72, f"all records {overall:.0%}", fontsize=7.5, color=INK_2, va="center")
    for yi, v, c in zip(y, male, CLASSES):
        ax.text(0.02, yi, f"{v:.0%}", va="center", ha="left", fontsize=7.5,
                color="white" if c in ("NORM", "MI") else INK)
    ax.set_ylim(-1.05, 4.5)
    ax.set_yticks(y, CLASSES, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    vgrid(ax)
    ax.spines["left"].set_visible(False)
    ax.set_title("(b) Share of male patients")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig05_demographics.png")
    summary["median_age"] = {c: float(np.median(d)) for c, d in zip(CLASSES, data)}
    summary["male_share_per_class"] = {c: round(v, 3) for c, v in zip(CLASSES, male)}


def fig_quality(df, summary):
    """Signal-quality flags, label certainty and human validation per fold."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1.25, 1, 1]})

    ax = axes[0]
    prev = {v: df[k].notna().mean() for k, v in QUALITY_FLAGS.items()}
    y = np.arange(len(prev))[::-1]
    ax.barh(y, list(prev.values()), height=0.58, color="#3987e5")
    for yi, v in zip(y, prev.values()):
        ax.text(v + 0.004, yi, f"{v:.1%}", va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(y, list(prev), fontsize=8, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.21)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    vgrid(ax)
    ax.spines["left"].set_visible(False)
    ax.set_title(f"(a) Quality flags ({df.any_quality_flag.mean():.0%} have one)")

    ax = axes[1]
    diag_codes = set(pd.read_csv(DATA / "scp_statements.csv", index_col=0).query("diagnostic == 1").index)
    lik = pd.Series([v for c in df.scp_codes for k, v in c.items() if k in diag_codes])
    levels = [100, 80, 50, 35, 15, 0]
    shares = [(lik == lv).mean() for lv in levels]
    other = 1 - sum(shares)
    ax.bar(range(len(levels)), shares, width=0.62, color="#3987e5")
    for i, v in enumerate(shares):
        ax.text(i, v + 0.015, f"{v:.0%}" if v >= 0.01 else f"{v:.1%}", ha="center", fontsize=7.5, color=INK_2)
    ax.set_xticks(range(len(levels)), [f"{lv}" for lv in levels])
    ax.set_xlabel("likelihood of the statement (%; 0 = unknown)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylim(0, 0.85)
    hgrid(ax)
    ax.set_title("(b) Certainty of diagnostic labels")

    ax = axes[2]
    val = df.groupby("strat_fold").validated_by_human.mean()
    colors = ["#c3c2b7" if f <= 8 else ACCENT for f in val.index]
    ax.bar(val.index, val.values, width=0.62, color=colors)
    ax.text(9.5, 1.03, "val / test", ha="center", fontsize=7.5, color=INK_2)
    ax.set_xticks(val.index)
    ax.set_xlabel("stratified fold")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylim(0, 1.12)
    hgrid(ax)
    ax.set_title("(c) Human-validated reports")
    fig.tight_layout(w_pad=1.3)
    save(fig, "fig06_quality_and_labels.png")
    summary["quality_flag_share"] = {k: round(float(v), 4) for k, v in prev.items()}
    summary["any_quality_flag_share"] = round(float(df.any_quality_flag.mean()), 4)
    summary["likelihood_share"] = {str(lv): round(float(s), 4) for lv, s in zip(levels, shares)}
    summary["likelihood_other_share"] = round(float(other), 4)
    summary["validated_by_human_per_fold"] = {int(k): round(float(v), 3) for k, v in val.items()}


def fig_leads(X, summary, n=4000, seed=0):
    """Average between-lead correlation, and how exactly the limb leads are redundant."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    corr = np.zeros((12, 12))
    for i in idx:
        corr += np.nan_to_num(np.corrcoef(highpass(X[i]).T))
    corr /= len(idx)

    I, II, III, aVR, aVL, aVF = (X[:, :, k].astype(np.float64) for k in range(6))
    def rel_err(target, approx):
        num = np.linalg.norm(target - approx, axis=1)
        den = np.linalg.norm(target, axis=1) + 1e-9
        return float(np.median(num / den))
    redundancy = {
        "III = II - I": rel_err(III, II - I),
        "aVR = -(I + II) / 2": rel_err(aVR, -(I + II) / 2),
        "aVL = I - II / 2": rel_err(aVL, I - II / 2),
        "aVF = II - I / 2": rel_err(aVF, II - I / 2),
    }

    fig, ax = plt.subplots(figsize=(3.6, 3.15))
    im = ax.imshow(corr, cmap=DIV, vmin=-1, vmax=1)
    for i in range(12):
        for j in range(12):
            if i != j and abs(corr[i, j]) >= 0.6:
                ax.text(j, i, f"{corr[i, j]:.1f}".replace("0.", "."), ha="center", va="center",
                        fontsize=6, color="white" if abs(corr[i, j]) > 0.75 else INK)
    ax.set_xticks(range(12), LEADS, fontsize=7.5, rotation=90)
    ax.set_yticks(range(12), LEADS, fontsize=7.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(5.5, color="white", linewidth=2)
    ax.axvline(5.5, color="white", linewidth=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, ticks=[-1, 0, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, length=0)
    ax.set_title("Mean correlation between leads")
    fig.tight_layout()
    save(fig, "fig07_lead_correlation.png")
    summary["limb_lead_median_relative_error"] = {k: round(v, 4) for k, v in redundancy.items()}
    summary["lead_correlation_sample"] = int(len(idx))


def fig_fold_balance(df, summary):
    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    prev = df.groupby("strat_fold")[CLASSES].mean()
    for c in CLASSES:
        ax.plot(prev.index, prev[c], color=CLASS_COLORS[c], linewidth=2, marker="o", markersize=4,
                markeredgecolor="white", markeredgewidth=1, label=c)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, handlelength=1.4)
    ax.set_xticks(prev.index)
    ax.set_xlim(0.7, 10.3)
    ax.set_ylim(0, 0.5)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("stratified fold (1-8 train, 9 validation, 10 test)")
    hgrid(ax)
    ax.set_title("Class prevalence per fold: the official split is well stratified")
    fig.tight_layout()
    save(fig, "fig08_fold_balance.png")
    summary["fold_prevalence_range_pct_points"] = {
        c: round(float(prev[c].max() - prev[c].min()) * 100, 2) for c in CLASSES}


def fig_signal_stats(df, X, summary):
    """Per-lead amplitude spread, which motivates per-lead z-scoring."""
    std = X.std(axis=1)  # (n_records, 12)
    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    q = np.percentile(std, [5, 25, 50, 75, 95], axis=0)
    xs = np.arange(12)
    ax.vlines(xs, q[0], q[4], color="#86b6ef", linewidth=2)
    ax.vlines(xs, q[1], q[3], color="#2a78d6", linewidth=7)
    ax.scatter(xs, q[2], s=30, color=INK, zorder=3, edgecolors="white", linewidths=1)
    ax.set_xticks(xs, LEADS)
    ax.set_ylabel("per-record std (mV)")
    ax.set_ylim(0, None)
    hgrid(ax)
    ax.set_title("Signal spread per lead (median, 25-75% and 5-95% of records)")
    fig.tight_layout()
    save(fig, "fig09_lead_amplitude.png")
    flat = (std < 1e-3).any(axis=1)
    summary["records_with_flat_lead"] = int(flat.sum())
    summary["median_std_per_lead_mV"] = {l: round(float(v), 3) for l, v in zip(LEADS, q[2])}
    summary["abs_amplitude_p99_9_mV"] = round(float(np.percentile(np.abs(X), 99.9)), 2)
    summary["abs_amplitude_max_mV"] = round(float(np.abs(X).max()), 2)


def slide_label_bars(df):
    """Records per superclass for the dataset slide (fits the left card)."""
    fig, ax = plt.subplots(figsize=(4.6, 2.65))
    stacked_class_bars(ax, df, label_size=10.5, legend=False)
    ax.legend(handles=[Patch(color="#6b6a66", label="only this label"),
                       Patch(color="#6b6a66", alpha=0.32, label="with other labels")],
              loc="lower right", fontsize=9.5, handlelength=1.2)
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    save(fig, "slide_label_bars.png", OUT_SLIDES)


def basic_summary(df):
    alone, total = alone_vs_shared(df)
    per_patient = df.patient_id.value_counts()
    return {
        "records": int(len(df)),
        "patients": int(df.patient_id.nunique()),
        "patients_with_multiple_records": int((per_patient > 1).sum()),
        "max_records_per_patient": int(per_patient.max()),
        "records_per_class": dict(zip(CLASSES, total)),
        "share_per_class": {c: round(t / len(df), 4) for c, t in zip(CLASSES, total)},
        "only_this_label": dict(zip(CLASSES, alone)),
        "only_this_label_share": {c: round(a / t, 3) for c, a, t in zip(CLASSES, alone, total)},
        "labels_per_record": {int(k): int(v) for k, v in df.n_labels.value_counts().sort_index().items()},
        "records_without_superclass": int((df.n_labels == 0).sum()),
        "norm_to_hyp_ratio": round(total[0] / total[-1], 2),
        "age_median": float(df.age.median()),
        "age_range": [float(df.age.min()), float(df.age.max())],
        "male_share": round(float((df.sex == 0).mean()), 3),
        "recording_years": [df.recording_date.min()[:4], df.recording_date.max()[:4]],
        "second_opinion": int(df.second_opinion.sum()),
        "validated_by_human": int(df.validated_by_human.sum()),
        "records_per_fold": {int(k): int(v) for k, v in df.strat_fold.value_counts().sort_index().items()},
        "height_available": round(float(df.height.notna().mean()), 3),
        "weight_available": round(float(df.weight.notna().mean()), 3),
    }


def main():
    df, scp = load_metadata()
    summary = basic_summary(df)
    print(f"{len(df):,} records from {df.patient_id.nunique():,} patients")

    fig_labels(df, summary)
    fig_label_combinations(df)
    fig_cooccurrence(df, scp, summary)
    fig_demographics(df, summary)
    fig_quality(df, summary)
    fig_fold_balance(df, summary)
    slide_label_bars(df)

    print("loading 100 Hz signals ...")
    X = load_signals(df)
    fig_example_ecg(df, X, summary)
    fig_median_beats(df, X, summary)
    fig_leads(X, summary)
    fig_signal_stats(df, X, summary)

    (ROOT / "eda").mkdir(exist_ok=True)
    with open(ROOT / "eda" / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote eda/eda_summary.json")


if __name__ == "__main__":
    main()
