# ECG Classification

Multi-label classification of 12-lead ECGs from the [PTB-XL](https://physionet.org/content/ptb-xl/) dataset [[1, 2]](#references) into the 5 diagnostic superclasses:

| <small>Code</small> | <small>Meaning</small> |
|:----:|:--------|
| <small>**NORM**</small> | <small>Normal ECG</small> |
| <small>**MI**</small> | <small>Myocardial infarction</small> |
| <small>**STTC**</small> | <small>ST/T change</small> |
| <small>**CD**</small> | <small>Conduction disturbance</small> |
| <small>**HYP**</small> | <small>Hypertrophy</small> |

Several classes can be true at once, so every model has 5 sigmoid outputs trained with binary cross-entropy.

---

## Contents

- [Research questions](#research-questions)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Data](#data)
- [Exploratory data analysis](#exploratory-data-analysis)
- [Standard training protocol](#standard-training-protocol)
- [Running experiments](#running-experiments)
- [Baseline](#baseline)
  - [B0: Can we reproduce a benchmark?](#b0-can-we-reproduce-a-benchmark)
- [Experiments](#experiments)
  - [E1: Does depth help?](#e1-does-depth-help-resnet1d-10-vs-resnet1d-18-vs-resnet1d-34)
  - [E2: Which operation type fits ECG?](#e2-which-operation-type-fits-ecg-convolution-vs-bilstm-vs-cnn--transformer)
  - [E3: Which augmentations help?](#e3-which-augmentations-help-none-vs-noise-baseline-wander-scaling-time-shift-random-crop)
  - [E4: How many leads are needed?](#e4-how-many-leads-are-needed-12-leads-vs-6-limb-leads-vs-lead-ii-only)
  - [E5: Does finer resolution help?](#e5-does-finer-resolution-help-100-hz-vs-500-hz)
- [Supplementary study: Optimizing the ResNet architecture](#supplementary-study-optimizing-the-resnet-architecture)
  - [S1: Which ResNet design choices help?](#s1-which-resnet-design-choices-help-custom-resnet-variants)
  - [S2: How far can a ResNet go?](#s2-how-far-can-a-resnet-go-architecture-search-within-the-resnet-family)
- [References](#references)

---

## Research questions

**Goal:** find which architecture reads ECGs best, and which choices drive its performance.

| <small>ID</small> | <small>Question</small> | <small>Variants compared</small> |
|:--:|:---------|:------------------|
| <small>**B0**</small> | <small>[Can we reproduce a benchmark?](#b0-can-we-reproduce-a-benchmark)</small> | <small>`resnet1d_wang` with the benchmark's training procedure; FCN (`fcn_wang`) with the standard protocol; RBF SVM on hand-crafted features and on the raw waveform; official split (train 1–8, val 9, test 10)</small> |
| <small>**E1**</small> | <small>[Does depth help?](#e1-does-depth-help-resnet1d-10-vs-resnet1d-18-vs-resnet1d-34)</small> | <small>ResNet1D-10 vs ResNet1D-18 vs ResNet1D-34</small> |
| <small>**E2**</small> | <small>[Which operation type fits ECG?](#e2-which-operation-type-fits-ecg-convolution-vs-bilstm-vs-cnn--transformer)</small> | <small>Convolution vs BiLSTM vs CNN + Transformer</small> |
| <small>**E3**</small> | <small>[Which augmentations help?](#e3-which-augmentations-help-none-vs-noise-baseline-wander-scaling-time-shift-random-crop)</small> | <small>None, noise, baseline wander, scaling, time shift, random crop</small> |
| <small>**E4**</small> | <small>[How many leads are needed?](#e4-how-many-leads-are-needed-12-leads-vs-6-limb-leads-vs-lead-ii-only)</small> | <small>12 leads vs 6 limb leads vs lead II only</small> |
| <small>**E5**</small> | <small>[Does finer resolution help?](#e5-does-finer-resolution-help-100-hz-vs-500-hz)</small> | <small>100 Hz (1000 × 12) vs 500 Hz (5000 × 12) input</small> |

**Supplementary study: Optimizing the ResNet architecture.** The ResNet received considerably more design effort than the other model families; this study is therefore reported separately from the research questions ([details](#supplementary-study-optimizing-the-resnet-architecture)).

| <small>ID</small> | <small>Question</small> | <small>Variants compared</small> |
|:--:|:---------|:------------------|
| <small>**S1**</small> | <small>[Which ResNet design choices help?](#s1-which-resnet-design-choices-help-custom-resnet-variants)</small> | <small>Classic ResNet1D-10 vs custom variants (stride-1 stem, concat-pooling head, constant width), deeper classic ResNets, `resnet1d_wang`</small> |
| <small>**S2**</small> | <small>[How far can a ResNet go?](#s2-how-far-can-a-resnet-go-architecture-search-within-the-resnet-family)</small> | <small>A single ResNet: architecture variants within the ResNet family, standard protocol unchanged, selected on validation</small> |

---

## Repository structure

```text
.
├── src/
│   ├── config.py                          paths, classes, lead sets, official split, STANDARD training protocol
│   ├── data/
│   │   ├── preprocess.py                  build the signal cache (once per sampling rate)
│   │   ├── dataset.py                     label aggregation, lead selection, split, standardization
│   │   └── augment.py                     noise / baseline wander / scaling / time shift / random crop (for E3)
│   ├── features/
│   │   └── ecg_features.py                hand-crafted per-lead features: statistical, ECG + wavelet (B0 SVM)
│   ├── models/
│   │   ├── __init__.py                    model registry
│   │   ├── SVM*.ipynb                     original SVM notebooks, kept for reference (B0)
│   │   ├── bilstm.py                      bidirectional LSTM on the raw signal (E2)
│   │   ├── cnn_transformer.py             convolutional front-end + Transformer encoder (E2)
│   │   ├── fcn1d.py                       FCN of Wang et al. (B0 baseline)
│   │   ├── heads.py                       classification heads (average pooling; concat pooling + dropout)
│   │   ├── resnet1d.py                    classic ResNet1D-10/-18/-34 and custom ResNet1D-10 variants (S1)
│   │   ├── resnet1d_plus.py               configurable ResNet for the architecture search (S2)
│   │   └── resnet1d_wang.py               resnet1d_wang of the PTB-XL benchmark (B0)
│   ├── training/
│   │   ├── svm_fit.py                     one binary SVM on a precomputed kernel (worker of train_svm.py)
│   │   ├── train.py                       training loop, logging, checkpointing (+ test evaluation at the end)
│   │   └── train_svm.py                   RBF SVM baseline: inputs, GPU kernel, grid on validation, evaluation
│   ├── evaluation/
│   │   ├── age_analysis.py                test metrics per age group and per sex (error analysis)
│   │   ├── evaluate.py                    (re-)evaluate runs: val/test metrics, val-tuned thresholds, bootstrap CIs
│   │   ├── metrics.py                     AUROC, AUPRC, F1 (macro + per class)
│   │   └── summarize.py                   collect runs of an experiment into tables (mean ± std over seeds)
│   ├── visualize/
│   │   ├── exploratory_data_analysis.py   EDA figures + summary numbers
│   │   └── plot_runs.py                   learning curves and per-class plots for any experiment
│   └── experiments/
│       ├── b0_baseline.py                 B0: FCN baseline under the standard protocol
│       ├── b0_benchmark.py                B0: reproduction of resnet1d_wang (Strodthoff et al.)
│       ├── b0_svm.py                      B0: SVM on statistical features, ECG + wavelet features, raw waveform
│       ├── e1_depth.py                    E1: ResNet1D-10 vs ResNet1D-18 vs ResNet1D-34
│       ├── e2_operation.py                E2: convolution vs BiLSTM vs CNN + Transformer
│       ├── e3_augment.py                  E3: none / noise / baseline wander / scaling / time shift / random crop / all combined
│       ├── e4_leads.py                    E4: 12 leads vs 6 limb leads vs lead II (ResNet1D-10)
│       ├── s1_resnet_design.py            S1: classic vs custom ResNet variants
│       └── s2_resnet_search.py            S2: ResNet architecture search (single model)
├── runs/<exp>/<run>/                      one folder per training run (weights not tracked)
├── eda/                                   EDA outputs
├── documentation/                         initial report (PDF) and presentation (PPTX)
└── data/                                  PTB-XL raw + processed (not tracked, see below)
```

> Each file that can be run has its usage in its docstring and `--help`.

---

## Setup

**Python 3.12.** PyTorch is installed separately because it needs the CUDA build (cu128 is required for RTX 50xx GPUs; pick another index for older GPUs or CPU).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

---

## Data

Download PTB-XL **v1.0.1** [[2]](#references) from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.1/) [[3]](#references) (21,837 records, the version used by the benchmark of Strodthoff et al. [[4]](#references); v1.0.3 removed duplicates and has 21,799) and extract it so that the files sit under `data/ptb-xl/` (e.g. `data/ptb-xl/ptbxl_database.csv`). Then build the cache:

```bash
python -m src.data.preprocess              # 100 Hz, ~1 GB, ~30 s
python -m src.data.preprocess --fs 500     # 500 Hz, ~5.2 GB (only needed for E5)
```

The cache stores **all** records with **all** 12 leads as raw mV (`data/processed/ptbxl_<fs>hz.npy`, shape `(21837, 12, fs*10)`, plus `labels.csv`). Everything that changes between experiments is applied at load time, so the same cache serves every experiment:

| <small>Option</small> | <small>Values</small> | <small>Used by</small> |
|:-------|:-------|:-------:|
| <small>`--fs`</small> | <small>`100`, `500`</small> | <small>E5</small> |
| <small>`--leads`</small> | <small>`all12`, `limb6`, `precordial6`, `lead1`, `lead2`, or any list like `I,II,V2`</small> | <small>E4</small> |
| <small>`--augment`</small> | <small>`none` or any of `noise,wander,scale,shift,crop`</small> | <small>E3</small> |
| <small>`--model`</small> | <small>`fcn1d`, `resnet1d10`, `resnet1d18`, `resnet1d34`, `resnet1d_wang`, and the S1 variants `resnet1d10_stem1`, `resnet1d10_chead`, `resnet1d10_w128`, `resnet1d10_stem1_chead_w128`, `resnet1d_plus`, `bilstm` and `cnn_transformer` (configured with `--model-kwargs`) (add new ones in `src/models/__init__.py`)</small> | <small>B0, E1, E2, S1, S2</small> |
| <small>`--standardize`</small> | <small>`per_lead` (standard) or `global` (one mean/std, as in the benchmark)</small> | <small>B0</small> |

**Labels:** every diagnostic SCP statement is mapped to its superclass regardless of likelihood (as in the PhysioNet example and the Strodthoff et al. benchmark [[4]](#references)). The 407 records with no superclass label are dropped. Signals are z-scored per lead with mean/std from the training folds only.

---

<details>
<summary><a id="exploratory-data-analysis"></a><h2 style="display:inline">Exploratory data analysis</h2></summary>

```bash
python -m src.visualize.exploratory_data_analysis
```

This writes the figures to `eda/figures/` (versions for slides to `eda/figures/slides/`) and the summary statistics to `eda/eda_summary.json`. All numbers refer to the 21,837 records of PTB-XL v1.0.1.

#### Labels

- **The classes are imbalanced.** NORM covers 9,528 records (44%), MI 5,486 (25%), STTC 5,250 (24%), CD 4,907 (22%) and HYP 2,655 (12%); NORM is 3.6 times as frequent as HYP.
- **The task is multi-label.** 75% of the records carry one superclass, 19% two, 4% three and 0.7% four; 407 records (1.9%) carry none and are dropped.
- **Pathologies rarely occur alone.** 95% of the NORM records carry no other superclass, but only 46% of the MI and STTC, 35% of the CD and 20% of the HYP records. 57% of the HYP records also carry STTC, mostly through the subclass pair ISC_ + LVH (1,172 records, 92% of all ISC_ records).

<p align="center">
  <img src="eda/figures/fig01_label_structure.png" alt="Records per superclass, superclasses per record and conditional co-occurrence" width="100%">
</p>

*Figure 1. Records per superclass (dark: only this superclass), superclasses per record, and the share of records with the row superclass that also carry the column superclass.*

<p align="center">
  <img src="eda/figures/fig02_label_combinations.png" alt="Most frequent exact label combinations" width="85%">
</p>

*Figure 2. The most frequent exact combinations of superclasses (logarithmic scale).*

<p align="center">
  <img src="eda/figures/fig10_cooccurrence_superclass.png" alt="Records carrying both superclasses" width="45%">
</p>

*Figure 3. Records carrying both superclasses; the diagonal counts the records with only this superclass.*

<p align="center">
  <img src="eda/figures/fig11_cooccurrence_subclass.png" alt="Co-occurrence of the diagnostic subclasses" width="100%">
</p>

*Figure 4. Co-occurrence of the 23 diagnostic subclasses; the diagonal gives the number of records per subclass, lines separate the superclasses.*

<p align="center">
  <img src="eda/figures/fig12_top_overlapping_pairs.png" alt="Most frequent overlapping diagnostic subclasses" width="90%">
</p>

*Figure 5. The most frequent pairs of diagnostic subclasses; pairs across superclasses (orange) make the task multi-label.*

#### Signals

- **The classes differ visibly in the average heartbeat.** In single-label records without artefacts:
  - HYP has a much larger R wave in V5 (median beat 2.10 mV, compared with 1.26 mV for NORM) and a deeper S wave in V2;
  - STTC has flattened T waves;
  - CD and MI have smaller R waves in V5 (0.73 and 0.93 mV).
- **The six limb leads carry only two independent signals.** III, aVR, aVL and aVF are linear combinations of I and II (Einthoven's and Goldberger's equations), reproduced in the data with a median relative error below 0.6%. The six limb leads of E4 therefore contain the same information as leads I and II.
- **The precordial leads have larger amplitudes.** Their median per-record standard deviation is 0.16–0.25 mV, compared with 0.10–0.13 mV for the limb leads, which motivates standardizing every lead separately. Absolute amplitudes stay below 2.2 mV in 99.9% of the samples, with outliers up to 20 mV.

<p align="center">
  <img src="eda/figures/fig03_example_ecg.png" alt="Example 12-lead record" width="100%">
</p>

*Figure 6. An example record (NORM): the 12 leads in the standard layout and lead II over the full 10 s.*

<p align="center">
  <img src="eda/figures/fig04_median_beats.png" alt="Median heartbeat per superclass" width="100%">
</p>

*Figure 7. Median heartbeat per superclass in leads II, V2 and V5, aligned at the R peak.*

<p align="center">
  <img src="eda/figures/fig07_lead_correlation.png" alt="Mean correlation between leads" width="45%">
</p>

*Figure 8. Mean correlation between the 12 leads.*

<p align="center">
  <img src="eda/figures/fig09_lead_amplitude.png" alt="Signal spread per lead" width="100%">
</p>

*Figure 9. Per-record standard deviation of each lead (median, 25–75% and 5–95% of the records).*

#### Patients, recording quality and split

- **Patients.** The records come from 18,885 patients (2,127 with several records, at most 10), recorded between 1984 and 2001. The median age is 62 years (range 2–95), and 52% of the patients are male. NORM patients are younger (median 54 years, compared with 67–68 for the other superclasses). MI, CD and HYP are more frequent in men (57–62% male, compared with 46% for NORM).
- **Recording quality.** 30% of the records carry a quality flag: static noise (14.9%), extra beats (8.9%), baseline drift (7.4%), burst noise (2.8%), pacemaker (1.3%) and electrode problems (0.1%). Noise and baseline wander are therefore realistic augmentations (E3).
- **Label certainty.** 74% of the diagnostic statements have a likelihood of 100%; the others range from 80% down to 15% or are unspecified, a source of label noise.
- **Split.** Folds 9 and 10 (validation and test) contain only reports validated by a cardiologist, compared with about 67% in folds 1–8, so the evaluation labels are of higher quality than the training labels. Class prevalence varies by at most 2.6 percentage points between folds.

<p align="center">
  <img src="eda/figures/fig05_demographics.png" alt="Age and sex per superclass" width="100%">
</p>

*Figure 10. Age and share of male patients per superclass.*

<p align="center">
  <img src="eda/figures/fig06_quality_and_labels.png" alt="Quality flags, label certainty and human validation" width="100%">
</p>

*Figure 11. Quality flags, certainty of the diagnostic statements, and share of human-validated reports per fold.*

<p align="center">
  <img src="eda/figures/fig08_fold_balance.png" alt="Class prevalence per fold" width="100%">
</p>

*Figure 12. Class prevalence per fold of the official split.*

</details>

---

## Standard training protocol

All experiments use these settings (defined once in `src/config.py`, `STANDARD`) and change only the factor they study.

| <small>Setting</small> | <small>Value</small> |
|:--------|:------|
| <small>**Split**</small> | <small>official `strat_fold`: train 1–8 (17,111 records), val 9 (2,156), test 10 (2,163)</small> |
| <small>**Input**</small> | <small>100 Hz, 12 leads, full 10 s, per-lead standardization</small> |
| <small>**Loss**</small> | <small>`BCEWithLogits`, 5 outputs</small> |
| <small>**Optimizer**</small> | <small>AdamW [[5]](#references), weight decay 0.01, batch size 128</small> |
| <small>**Schedule**</small> | <small>one-cycle [[6]](#references), peak LR 1e-3, 30% warm-up, cosine decay</small> |
| <small>**Epochs**</small> | <small>**50, fixed** (no early stopping)</small> |
| <small>**Checkpoint**</small> | <small>epoch with the best **validation macro AUROC**</small> |
| <small>**Seeds**</small> | <small>0, 1, 2; results reported as mean ± std</small> |
| <small>**Test metrics**</small> | <small>macro + per-class AUROC and AUPRC; macro + per-class F1 at 0.5 and at per-class thresholds tuned on validation; 95% percentile bootstrap CIs [[7]](#references) (1,000 resamples)</small> |
| <small>**Precision**</small> | <small>bf16 autocast on GPU; runs are deterministic for a given seed and GPU</small> |

---

## Running experiments

```bash
python -m src.training.train --exp scratch --model resnet1d18 --seeds 0   # one run
python -m src.training.train --exp e4_leads --model resnet1d18 --leads limb6 --seeds 0 1 2
python -m src.training.train_svm --exp b0_svm --features wavelet   # one SVM run (B0)
python -m src.evaluation.evaluate --exp e1_depth     # re-evaluate + summarize
python -m src.evaluation.summarize --exp e1_depth
python -m src.visualize.plot_runs --exp e1_depth
```

Each run writes to `runs/<exp>/<model>_<fs>hz_<leads>[_aug-…]_s<seed>/`:

| <small>File</small> | <small>Content</small> |
|:-----|:--------|
| <small>`config.json`</small> | <small>every setting, parameter count, dataset sizes, git commit, best epoch, timing</small> |
| <small>`history.csv`</small> | <small>one row per epoch: `lr`, `train_loss`, `val_loss`, and `{train,val}_{macro_auc,macro_ap,macro_f1}`, `{train,val}_{auc,ap,f1}_<CLASS>`</small> |
| <small>`metrics.json`</small> | <small>val and test metrics of the best checkpoint, tuned thresholds, CIs</small> |
| <small>`metrics_last.json`</small> | <small>the same for the last-epoch checkpoint (only if evaluated with `--checkpoint last`, e.g. in B0)</small> |
| <small>`predictions.npz`</small> | <small>val/test `ecg_id`, labels and probabilities (for ROC/PR curves)</small> |
| <small>`norm_stats.npz`</small> | <small>per-lead mean/std used for standardization</small> |
| <small>`best.pt`, `last.pt`</small> | <small>weights (not tracked by git)</small> |

Per experiment: `runs/<exp>/runs.csv`, `summary.csv` / `summary.md` (mean ± std per variant) and `figures/`.

SVM runs (`src/training/train_svm.py`) write `config.json`, `metrics.json` and `predictions.npz` in the same format, so summaries, figures and `age_analysis` work unchanged. They have no `history.csv`, normalization statistics or weights; `config.json` holds the selected C and γ and the validation score of every grid point.

---

## Baseline

<details>
<summary><a id="b0-can-we-reproduce-a-benchmark"></a><h3 style="display:inline">B0: Can we reproduce a benchmark?</h3></summary>

```bash
python -m src.experiments.b0_benchmark                  # resnet1d_wang, benchmark procedure, 3 seeds
python -m src.experiments.b0_baseline                   # FCN, standard protocol, 3 seeds
python -m src.evaluation.age_analysis --exp b0_baseline # performance by age and sex
python -m src.experiments.b0_svm                        # SVM: statistical, wavelet and raw inputs
```

B0 comprises three baselines. The first two are networks of Wang et al. [[10]](#references), both part of the PTB-XL benchmark of Strodthoff et al. [[4]](#references); the third is a classical, non-deep model:

1. `resnet1d_wang`, trained with the benchmark's training procedure, to verify that the pipeline reproduces a published result;
2. the fully convolutional network (FCN), trained with the standard protocol, as a simple convolutional baseline for the subsequent experiments, together with an analysis of performance by age and sex.
3. a support vector machine (SVM) on hand-crafted features and on the raw waveform, to quantify what learning from the signal adds over a classical feature-based approach.

#### Reproduction of `resnet1d_wang`

The `resnet1d_wang` model of the PTB-XL benchmark of Strodthoff et al. [[4]](#references), the 1D ResNet of Wang et al. [[10]](#references) as implemented in the benchmark code, is reimplemented in `src/models/resnet1d_wang.py`: a stem convolution (kernel size 7, stride 1, no pooling), three residual blocks of 128 channels with kernel sizes 5 and 3 and strides 1, 2 and 2, and a head of concatenated max and average pooling, batch-norm, dropout 0.25, a linear layer of 128 units, ReLU, batch-norm, dropout 0.5 and the output layer (0.47M parameters). It is trained with the training procedure of the benchmark, which differs from the standard protocol in the following points:

| <small>Setting</small> | <small>Benchmark procedure (B0)</small> | <small>Standard protocol</small> |
|:--------|:-------------------------|:------------------|
| <small>**Training input**</small> | <small>random 2.5 s crop of each record</small> | <small>full 10 s record</small> |
| <small>**Evaluation**</small> | <small>7 windows of 2.5 s (stride 1.25 s), per-class maximum</small> | <small>full 10 s record</small> |
| <small>**Peak learning rate**</small> | <small>1e-2</small> | <small>1e-3</small> |
| <small>**Adam β₂**</small> | <small>0.99</small> | <small>0.999</small> |
| <small>**Standardization**</small> | <small>one global mean/std over all leads</small> | <small>one mean/std per lead</small> |
| <small>**Checkpoint**</small> | <small>final epoch</small> | <small>epoch with the best validation macro AUROC</small> |

All other settings (100 Hz, 12 leads, official split, BCE loss, AdamW with weight decay 0.01, one-cycle schedule, 50 epochs, batch size 128) are shared. Both checkpoints are evaluated: the final epoch, which is the benchmark's procedure, and the best validation epoch, which is the selection used in all other experiments.

*resnet1d_wang, test set (fold 10), mean ± std over 3 seeds:*

| <small>Checkpoint</small> | <small>Macro AUROC</small> | <small>Macro AUPRC</small> | <small>Macro F1 @0.5</small> | <small>Macro F1 @tuned</small> | <small>NORM</small> | <small>MI</small> | <small>STTC</small> | <small>CD</small> | <small>HYP</small> |
|:-----------|------------:|------------:|--------------:|----------------:|-----:|---:|-----:|---:|----:|
| <small>Published value [[4]](#references)</small> | <small>$\color{#BF3E2A}{0.930}$</small> | | | | | | | | |
| <small>**Final epoch** (benchmark procedure)</small> | <small>$\color{#C0392B}{0.9299 ± 0.0010}$</small> | <small>$\color{#C0392B}{0.8292 ± 0.0034}$</small> | <small>$\color{#C0392B}{0.7510 ± 0.0032}$</small> | <small>$\color{#C0392B}{0.7530 ± 0.0057}$</small> | <small>0.9490</small> | <small>$\color{#C0392B}{0.9359}$</small> | <small>$\color{#C0392B}{0.9326}$</small> | <small>$\color{#C0392B}{0.9263}$</small> | <small>$\color{#C0392B}{0.9058}$</small> |
| <small>**Best validation epoch**</small> | <small>$\color{#1E8449}{0.9324 ± 0.0011}$</small> | <small>$\color{#1E8449}{0.8369 ± 0.0013}$</small> | <small>$\color{#1E8449}{0.7557 ± 0.0034}$</small> | <small>$\color{#1E8449}{0.7577 ± 0.0049}$</small> | <small>0.9490</small> | <small>$\color{#1E8449}{0.9395}$</small> | <small>$\color{#1E8449}{0.9346}$</small> | <small>$\color{#1E8449}{0.9295}$</small> | <small>$\color{#1E8449}{0.9093}$</small> |

(NORM … HYP: per-class test AUROC.)

- **The benchmark result is reproduced.** With the benchmark's training procedure and checkpoint, the macro AUROC is 0.9299 ± 0.0010, against the published 0.930. The three runs lie between 0.9291 and 0.9310, and the published value lies within the 95% bootstrap confidence interval of each run (0.922–0.938).

- **Selecting the checkpoint on the validation set adds 0.0025.** Validation macro AUROC reaches its maximum between epochs 32 and 35 and then declines slightly (from 0.935 to 0.932 at epoch 50), while the validation loss rises from 0.278 to 0.306. The best-validation checkpoint therefore reaches 0.9324 ± 0.0011. Since the selection uses only the validation fold, the test set remains untouched.

- **The training procedure prevents memorization of the training set.** At the end of training the training loss is 0.18, compared with approximately 0.0005 for the models trained with the standard protocol in E1, which is consistent with the effect of random cropping observed in E3.

- **Comparison with ResNet1D-10.** Trained with random crop in E3, ResNet1D-10 (2.8M parameters) reaches 0.9280 ± 0.0015; `resnet1d_wang` with the benchmark procedure reaches 0.9324 ± 0.0011 with the same checkpoint selection, six times fewer parameters and a similar training time (33 s per run). Since the two also differ in learning rate, β₂, standardization and architecture (no downsampling in the stem, dropout in the head), this comparison does not identify which of these differences accounts for the advantage.

- **Remaining differences from the original setup.** The reproduction uses the same data release as the benchmark (PTB-XL v1.0.1), but PyTorch instead of fastai v1, with the implementation details of each framework (e.g. default weight initialization). Given the agreement with the published value, these differences appear to have no material effect on the result.

<p align="center">
  <img src="runs/b0_benchmark/figures/curves.png" alt="B0 learning curves" width="100%">
</p>

#### FCN baseline under the standard protocol

The fully convolutional network (FCN) of Wang et al. [[10]](#references) (`src/models/fcn1d.py`) consists of three blocks of Conv1d + BatchNorm + ReLU with 128/256/128 filters and kernel sizes 8/5/3 (stride 1, "same" padding, no downsampling), followed by global average pooling and a linear layer (**0.28M** parameters). Its convolutional layers are those of `fcn_wang` in the benchmark of Strodthoff et al. [[4]](#references), which reports a macro AUROC of **0.925** on the superclass task. Two differences remain: the benchmark uses the same head as `resnet1d_wang` (concatenated max and average pooling and a hidden layer of 128 units), and the FCN here is trained with the standard protocol rather than with the benchmark's procedure. The comparison with the published value is therefore approximate.

`age_analysis` recomputes the test metrics of every run separately per age group (<40, 40–59, 60–79, 80+; the PTB-XL placeholder age 300 is mapped to 90+) and per sex, with group sizes, class prevalence per group and 95% bootstrap CIs. It only reads `predictions.npz`, so it also works on any earlier experiment (e.g. `--exp e1_depth`). Outputs: `runs/<exp>/subgroups/` (`age_summary.md`, `sex_summary.md`, CSVs, `figures/`).

*Test set (fold 10), mean ± std over 3 seeds:*

| <small>Model</small> | <small>Params</small> | <small>Macro AUROC</small> | <small>Macro AUPRC</small> | <small>Macro F1 @0.5</small> | <small>Macro F1 @tuned</small> | <small>NORM</small> | <small>MI</small> | <small>STTC</small> | <small>CD</small> | <small>HYP</small> |
|:------|-------:|------------:|------------:|--------------:|----------------:|-----:|---:|-----:|---:|----:|
| <small>**FCN (B0)**</small> | <small>0.28M</small> | <small>$\color{#C0392B}{0.9192 ± 0.0013}$</small> | <small>0.8065 ± 0.0012</small> | <small>0.7231 ± 0.0055</small> | <small>0.7368 ± 0.0035</small> | <small>0.9395</small> | <small>0.9232</small> | <small>0.9322</small> | <small>0.9063</small> | <small>0.8947</small> |
| <small>`fcn_wang`, benchmark [[4]](#references)</small> | <small>–</small> | <small>$\color{#1E8449}{0.925}$</small> | <small>–</small> | <small>–</small> | <small>–</small> | <small>–</small> | <small>–</small> | <small>–</small> | <small>–</small> | <small>–</small> |

- **Close to the published value.** The macro AUROC (0.919) is 0.006 below the 0.925 reported for `fcn_wang` [[4]](#references). The difference in training procedure is the most likely cause: random 2.5 s crops alone improve ResNet1D-10 by 0.016 (E3), and `resnet1d_wang` reproduces its published value when trained with the benchmark's procedure (above). The three seeds give 0.9184, 0.9185 and 0.9206, and re-running a seed reproduces its result exactly.

- **No overfitting.** The best validation epoch lies between 32 and 37 and the validation loss stays at approximately 0.29 until the end of training, while the training loss decreases to 0.19. With 0.28M parameters the network cannot memorize the training set, unlike the ResNets of E1.

- **The small FCN outperforms all three ResNets of E1** (0.919 vs 0.912–0.913) with 10–42 times fewer parameters, consistently over all seeds. Together with the early overfitting of the ResNets, this indicates that performance on PTB-XL is limited by generalization rather than by model capacity.

- **CD and HYP remain the weakest classes**, as for the ResNets.

#### Performance by age and sex

*Test set (fold 10), macro AUROC, mean ± std over 3 seeds, 95% bootstrap CI (B0); 15 test records without age are left out of the age groups:*

| <small>Age</small> | <small>n</small> | <small>% NORM</small> | <small>FCN (B0)</small> | <small>95% CI</small> | <small>ResNet1D-18</small> | <small>ResNet1D-34</small> |
|:----|--:|-------:|---------:|:------:|------------:|------------:|
| <small><40</small> | <small>284</small> | <small>83.1</small> | <small>$\color{#1E8449}{0.9225 ± 0.0054}$</small> | <small>[0.890, 0.949]</small> | <small>$\color{#7C7D2F}{0.8988}$</small> | <small>$\color{#5C8038}{0.9033}$</small> |
| <small>40–59</small> | <small>642</small> | <small>60.4</small> | <small>$\color{#46813E}{0.9159 ± 0.0019}$</small> | <small>[0.899, 0.931]</small> | <small>$\color{#41823F}{0.9076}$</small> | <small>$\color{#46813E}{0.9070}$</small> |
| <small>60–79</small> | <small>916</small> | <small>34.8</small> | <small>$\color{#6C7E34}{0.9099 ± 0.0011}$</small> | <small>[0.899, 0.921]</small> | <small>$\color{#5E7F37}{0.9031}$</small> | <small>$\color{#56803A}{0.9042}$</small> |
| <small>80+</small> | <small>306</small> | <small>6.9</small> | <small>$\color{#C0392B}{0.8734 ± 0.0010}$</small> | <small>[0.848, 0.895]</small> | <small>$\color{#C0392B}{0.8679}$</small> | <small>$\color{#C0392B}{0.8652}$</small> |
| <small>all</small> | <small>2163</small> | <small>44.6</small> | <small>$\color{#348243}{0.9192 ± 0.0013}$</small> | <small>[0.912, 0.926]</small> | <small>$\color{#1E8449}{0.9127}$</small> | <small>$\color{#1E8449}{0.9131}$</small> |

- **Performance drops for patients aged 80+** for all three models; for B0 the 95% CI of this group does not overlap with that of the whole test set. In this group only 6.9% of the records are normal and most carry several superclasses at once (MI 46.7%, STTC 42.8%, CD 42.2%), so co-occurring pathologies have to be separated. HYP is affected most (AUROC 0.818 for B0 at 80+).
- **Results for patients under 40 are uncertain:** the group contains few abnormal records (e.g. 7 MI cases), which gives a wide CI.
- **By sex**, B0 reaches 0.931 [0.921, 0.939] for men and 0.907 [0.895, 0.919] for women, with the largest gaps for MI (0.938 vs 0.905) and STTC (0.944 vs 0.919); the ResNets show the same direction (0.918 vs 0.907 for ResNet1D-18).
- Full tables: `runs/b0_baseline/subgroups/` and `runs/e1_depth/subgroups/`.

<p align="center">
  <img src="runs/b0_baseline/figures/curves.png" alt="B0 learning curves" width="100%">
</p>
<p align="center">
  <img src="runs/b0_baseline/subgroups/figures/age_auc.png" alt="B0 test AUROC by age" width="49%">
  <img src="runs/b0_baseline/subgroups/figures/age_auc_per_class.png" alt="B0 per-class test AUROC by age" width="49%">
</p>

#### Classical baseline: support vector machine

A support vector machine (SVM) with an RBF kernel [[21]](#references) is trained on three input representations, taken from the SVM notebooks of the project (`src/models/SVM*.ipynb`, kept for reference) and reimplemented in the standard pipeline (`src/features/ecg_features.py`, `src/training/train_svm.py`):

| <small>Input</small> | <small>Per lead</small> | <small>Inputs (12 leads)</small> | <small>Content</small> |
|:------|---------:|------------------:|:--------|
| <small>**Statistical features**</small> | <small>39</small> | <small>468</small> | <small>amplitude statistics and percentiles, distribution shape, first and second differences, Hjorth parameters, Welch power-spectrum descriptors; no beat detection</small> |
| <small>**ECG + wavelet features**</small> | <small>62</small> | <small>744</small> | <small>the same kind of statistics after band-pass filtering, approximate R-peaks and RR intervals, a median heartbeat summarized before, around and after the QRS complex, and a 4-level Daubechies-4 wavelet decomposition</small> |
| <small>**Raw waveform**</small> | <small>1,000</small> | <small>12,000</small> | <small>every sample of the 10 s record</small> |

**Model and selection.**
- Inputs are z-scored with training statistics. One binary SVM is trained per superclass (one-vs-rest, balanced class weights), all sharing C and γ.
- The grid, the same for all three inputs, is C ∈ {0.25, 1, 4, 16, 64} × γ ∈ {0.125, 0.25, …, 8} × γ<sub>scale</sub>, with γ<sub>scale</sub> = 1 / (n<sub>inputs</sub> · Var(X<sub>train</sub>)), scikit-learn's "scale". The configuration with the highest validation macro AUROC is selected; every selected configuration lies inside the grid.
- Class probabilities are the sigmoid of the decision function.
- The RBF kernel is computed on the GPU and passed to libsvm as a precomputed kernel. This is the same model, but fast enough for the 12,000-dimensional raw input (about 2 min per input for the whole grid).
- The SVM has no random component, so one run per input is reported, with 95% bootstrap confidence intervals; a second seed reproduced the predictions exactly.

**Differences from the original notebooks.** The SVM is trained and evaluated on the same records as all other models: PTB-XL v1.0.1, official split, records without a superclass label dropped. The notebooks used v1.0.3 and kept the unlabeled records. Selection uses macro AUROC instead of macro F1. The feature code is unchanged and gives identical values on 40 ECGs × 12 leads; only the wavelet decomposition (`pywt.wavedec`) was reimplemented in NumPy, because the compiled PyWavelets extension could not be loaded on the training machine.

*Test set (fold 10). SVM: one run per input, 95% bootstrap CI; FCN: mean ± std over 3 seeds.*

| <small>Model</small> | <small>Inputs</small> | <small>Macro AUROC</small> | <small>Macro AUPRC</small> | <small>Macro F1 @0.5</small> | <small>Macro F1 @tuned</small> | <small>NORM</small> | <small>MI</small> | <small>STTC</small> | <small>CD</small> | <small>HYP</small> |
|:------|-------:|------------:|------------:|--------------:|----------------:|-----:|---:|-----:|---:|----:|
| <small>SVM, statistical features</small> | <small>468</small> | <small>$\color{#56803A}{0.8884 [0.880, 0.897]}$</small> | <small>$\color{#5C8038}{0.7419}$</small> | <small>$\color{#368242}{0.6845}$</small> | <small>$\color{#677F35}{0.6816}$</small> | <small>$\color{#5A8039}{0.9117}$</small> | <small>$\color{#7F7D2E}{0.8561}$</small> | <small>$\color{#46813E}{0.9044}$</small> | <small>$\color{#56803A}{0.8703}$</small> | <small>$\color{#298346}{0.8995}$</small> |
| <small>**SVM, ECG + wavelet features**</small> | <small>744</small> | <small>$\color{#3B8241}{0.9031 [0.896, 0.911]}$</small> | <small>$\color{#41823F}{0.7696}$</small> | <small>$\color{#298346}{0.7068}$</small> | <small>$\color{#368242}{0.7171}$</small> | <small>$\color{#41823F}{0.9232}$</small> | <small>$\color{#697F34}{0.8702}$</small> | <small>$\color{#268347}{0.9264}$</small> | <small>$\color{#348243}{0.8931}$</small> | <small>$\color{#1E8449}{0.9027}$</small> |
| <small>SVM, raw waveform</small> | <small>12,000</small> | <small>$\color{#C0392B}{0.7488 [0.737, 0.761]}$</small> | <small>$\color{#C0392B}{0.4874}$</small> | <small>$\color{#C0392B}{0.2089}$</small> | <small>$\color{#C0392B}{0.4994}$</small> | <small>$\color{#C0392B}{0.7983}$</small> | <small>$\color{#C0392B}{0.7102}$</small> | <small>$\color{#C0392B}{0.7164}$</small> | <small>$\color{#C0392B}{0.7150}$</small> | <small>$\color{#C0392B}{0.8040}$</small> |
| <small>FCN (above)</small> | <small>–</small> | <small>$\color{#1E8449}{0.9192 ± 0.0013}$</small> | <small>$\color{#1E8449}{0.8065}$</small> | <small>$\color{#1E8449}{0.7231}$</small> | <small>$\color{#1E8449}{0.7368}$</small> | <small>$\color{#1E8449}{0.9395}$</small> | <small>$\color{#1E8449}{0.9232}$</small> | <small>$\color{#1E8449}{0.9322}$</small> | <small>$\color{#1E8449}{0.9063}$</small> | <small>$\color{#368242}{0.8947}$</small> |
| <small>`Wavelet+NN`, benchmark [[4]](#references)</small> | <small>–</small> | <small>$\color{#6E7E33}{0.874}$</small> | <small>–</small> | <small>–</small> | <small>–</small> | <small>–</small> | <small>–</small> | <small>–</small> | <small>–</small> | <small>–</small> |

- **Learning from the signal outperforms hand-crafted features.** The best SVM (ECG + wavelet features, 0.903) lies 0.016 below the FCN trained with the standard protocol. In a paired bootstrap on the test set, the FCN is better by 0.015–0.018 on each seed, with all 95% confidence intervals above zero. It lies 0.027 below `resnet1d_wang` (0.930). It is above the classical baseline of the benchmark (`Wavelet+NN`, 0.874) [[4]](#references).

- **ECG-specific features help.** Beat-level information (R-peaks, RR intervals, median heartbeat) and the wavelet decomposition raise macro AUROC by 0.015 over purely statistical features (paired bootstrap: 95% CI [0.010, 0.019]). The largest gains are for CD (+0.023), STTC (+0.022) and MI (+0.014).

- **The gap to the neural network is largest for MI, while HYP is on par.** For MI the SVM reaches 0.870, compared with 0.923 for the FCN; for HYP it reaches 0.903, compared with 0.895. A plausible explanation: hypertrophy is largely defined by voltage criteria, i.e. amplitudes that the features measure directly. Myocardial infarction, by contrast, shows in the shape of specific waves (Q waves, ST segment) in specific leads, which summary statistics capture only partly.

- **An SVM on the raw waveform does not generalize (0.749).** The RBF kernel compares two ECGs sample by sample, so the same heartbeat shifted by a fraction of a second appears dissimilar. Almost every training record becomes a support vector (11,380–16,850 of 17,111 per class): the model memorizes the training set rather than learning a decision rule. A representation is needed, either hand-crafted features or the learned, shift-equivariant filters of a convolutional network.

- **The results are consistent with the original notebooks.** Macro F1 at tuned thresholds is 0.682 for the statistical features and 0.499 for the raw waveform, compared with 0.675 and 0.496 in the notebooks on PTB-XL v1.0.3.

<p align="center">
  <img src="runs/b0_svm/figures/test_auc_per_class.png" alt="B0 SVM test AUROC per class" width="60%">
</p>

</details>

---

## Experiments

<details>
<summary><a id="e1-does-depth-help-resnet1d-10-vs-resnet1d-18-vs-resnet1d-34"></a><h3 style="display:inline">E1: Does depth help? ResNet1D-10 vs ResNet1D-18 vs ResNet1D-34</h3></summary>

```bash
python -m src.experiments.e1_depth
```

All three networks are the ResNet layout [[8]](#references) carried over to 1D (stem conv k=7 stride 2 + max-pool, four stages of basic blocks with 64/128/256/512 channels and kernel size 5, global average pooling, linear head). The last batch-norm of every residual branch is zero-initialized [[9]](#references), so each block starts as the identity and the deeper networks train as easily as the shallow one. They differ only in blocks per stage: `[1,1,1,1]` (**2.8M** parameters), `[2,2,2,2]` (**6.3M** parameters) and `[3,4,6,3]` (**11.9M** parameters). ResNet1D-10 is not one of the configurations of He et al.; it is the shallowest member of the same family, with a single block per stage.

#### Results

*Test set (fold 10), mean ± std over 3 seeds:*

| <small>Model</small> | <small>Params</small> | <small>Train time</small> | <small>Macro AUROC</small> | <small>Macro AUPRC</small> | <small>Macro F1 @0.5</small> | <small>Macro F1 @tuned</small> | <small>NORM</small> | <small>MI</small> | <small>STTC</small> | <small>CD</small> | <small>HYP</small> |
|:------|-------:|-----------:|------------:|------------:|--------------:|----------------:|-----:|---:|-----:|---:|----:|
| <small>**ResNet1D-10**</small> | <small>2.8M</small> | <small>33 s</small> | <small>$\color{#C0392B}{0.9121 ± 0.0007}$</small> | <small>$\color{#C0392B}{0.7960 ± 0.0016}$</small> | <small>$\color{#A17B25}{0.6996 ± 0.0231}$</small> | <small>$\color{#C0392B}{0.7232 ± 0.0045}$</small> | <small>$\color{#C0392B}{0.9350}$</small> | <small>$\color{#C0392B}{0.9020}$</small> | <small>$\color{#BA6223}{0.9270}$</small> | <small>$\color{#B96822}{0.9010}$</small> | <small>$\color{#1E8449}{0.8952}$</small> |
| <small>**ResNet1D-18**</small> | <small>6.3M</small> | <small>53 s</small> | <small>$\color{#9A7B27}{0.9127 ± 0.0017}$</small> | <small>$\color{#1E8449}{0.7978 ± 0.0036}$</small> | <small>$\color{#1E8449}{0.7049 ± 0.0160}$</small> | <small>$\color{#1E8449}{0.7255 ± 0.0066}$</small> | <small>$\color{#BA6523}{0.9356}$</small> | <small>$\color{#1E8449}{0.9049}$</small> | <small>$\color{#1E8449}{0.9283}$</small> | <small>$\color{#C0392B}{0.8995}$</small> | <small>$\color{#3B8241}{0.8951}$</small> |
| <small>**ResNet1D-34**</small> | <small>11.9M</small> | <small>94 s</small> | <small>$\color{#1E8449}{0.9131 ± 0.0017}$</small> | <small>$\color{#747E31}{0.7973 ± 0.0040}$</small> | <small>$\color{#C0392B}{0.6926 ± 0.0098}$</small> | <small>$\color{#BB6024}{0.7239 ± 0.0053}$</small> | <small>$\color{#1E8449}{0.9368}$</small> | <small>$\color{#53803A}{0.9044}$</small> | <small>$\color{#C0392B}{0.9264}$</small> | <small>$\color{#1E8449}{0.9036}$</small> | <small>$\color{#C0392B}{0.8942}$</small> |

(Train time: 50 epochs on an RTX 5080. NORM … HYP: per-class test AUROC.)

- **No measurable difference in performance.** Macro AUROC increases with depth only negligibly (0.9121 → 0.9127 → 0.9131); the full range across the three models (0.0010) is smaller than the standard deviation across seeds of the two deeper models (0.0017) and lies well within the 95% bootstrap confidence interval of each individual run (approximately ± 0.0075). The per-class AUROC values overlap for all five superclasses; the largest deviations, a slightly lower MI value for ResNet1D-10 (−0.003) and a slightly higher CD value for ResNet1D-34 (+0.004), both relative to ResNet1D-18, are within the range of random variation.

- **Increased computational cost without performance benefit.** ResNet1D-34 requires 4.2 times as many parameters and 2.8 times the training time of ResNet1D-10 while achieving equivalent performance. ResNet1D-10 is therefore adopted as the convolutional reference model for the subsequent experiments.

- **Early onset of overfitting in all three models.** Validation macro AUROC reaches its maximum between epochs 8 and 11. Beyond this point, the training loss decreases to approximately 0.0005 while the validation loss increases from 0.31 to between 0.77 and 0.84. Even the smallest network memorizes the training set: the shallower the network, the slightly more slowly this happens (training loss at epoch 20 of 0.113, 0.096 and 0.089, final validation loss of 0.77, 0.83 and 0.84 for ResNet1D-10, -18 and -34), but this does not translate into a higher validation AUROC. Within the range examined, generalization is therefore limited by insufficient regularization and training data rather than by model capacity, which motivates the augmentation study in E3.

- **Discrepancy with the published benchmark.** The macro AUROC obtained here (0.913) is below the 0.930 reported for `resnet1d_wang`, the 1D ResNet of Wang et al. [[10]](#references), in the PTB-XL benchmark of Strodthoff et al. [[4]](#references). Since reducing the depth to a single block per stage leaves performance unchanged, depth alone does not account for the difference to this considerably smaller network (≈ 0.5M parameters, three residual blocks of 128 channels). A plausible explanation is a difference in training procedure: the benchmark models are trained on randomly cropped 2.5 s segments and evaluated by aggregating predictions over sliding windows, which acts as a strong form of data augmentation, whereas the present models are trained on the full 10 s recordings without augmentation. This explanation is examined in E3.

<p align="center">
  <img src="runs/e1_depth/figures/curves.png" alt="E1 learning curves" width="100%">
</p>

</details>

<details>
<summary><a id="e2-which-operation-type-fits-ecg-convolution-vs-bilstm-vs-cnn--transformer"></a><h3 style="display:inline">E2: Which operation type fits ECG? Convolution vs BiLSTM vs CNN + Transformer</h3></summary>

```bash
python -m src.experiments.e2_operation                             # 6 models x 3 seeds
python -m src.experiments.e2_operation --variants lstm_small_fp16 lstm_large_fp16   # float16-LSTM check
python -m src.experiments.e2_operation --report-only --show-test   # final report with test set
```

The three model families differ only in the operation that processes the signal over time. Everything else is shared: the standard protocol with random crop (E3), the evaluation (maximum over the sliding windows), and the classifier on the pooled features (the MLP of the concatenated-pooling head, S1).

| <small>Family</small> | <small>Operation over time</small> | <small>Implementation</small> |
|:-------|:--------------------|:---------------|
| <small>**Convolution**</small> | <small>stacked local filters; the receptive field grows with depth</small> | <small>ResNet of S1/S2 (`resnet1d_plus`): stride-1 stem, four residual stages, downsampling ×8</small> |
| <small>**BiLSTM**</small> | <small>recurrence: the 12 lead values of each time step are read in sequence, forwards and backwards [[17]](#references) [[18]](#references); no convolutions</small> | <small>two stacked bidirectional LSTM layers on the raw signal, following the recurrent model of the benchmark [[4]](#references); outputs pooled by average, maximum and the final state of each direction; recurrence computed in float32 (see below)</small> |
| <small>**CNN + Transformer**</small> | <small>local convolutions, then self-attention between all positions [[19]](#references)</small> | <small>the ResNet stem and two strided residual blocks (63 tokens per 2.5 s window), sinusoidal position encoding, pre-norm Transformer encoder [[20]](#references)</small> |

**Sizes and selection.** Every family is trained in two sizes, small (0.6–0.8M parameters) and large (2.3–5.5M), so that a poorly chosen size does not decide the comparison.
- The large BiLSTM (hidden size 256, two layers) has the size of the benchmark's recurrent model.
- The ResNets are `base` of S2 and its wider and deeper variant (width 256, two blocks per stage, stochastic depth).
- The lead-wise stem selected in S2 is not used. Otherwise convolution would benefit from an architecture search that the other families did not receive.

Within each family the size with the higher mean validation macro AUROC is selected. The test set was evaluated only after this choice.

#### Results

*Mean ± std over 3 seeds; test set (fold 10). Selected size per family in bold. Train time: 50 epochs on an RTX 5080.*

| <small>Family</small> | <small>Model</small> | <small>Params</small> | <small>Train time</small> | <small>Val macro AUROC</small> | <small>Test macro AUROC</small> | <small>Macro AUPRC</small> | <small>Macro F1 @tuned</small> | <small>NORM</small> | <small>MI</small> | <small>STTC</small> | <small>CD</small> | <small>HYP</small> |
|:-------|:------|-------:|-----------:|----------------:|-----------------:|------------:|----------------:|-----:|---:|-----:|---:|----:|
| <small>Convolution</small> | <small>ResNet, small (`base`)</small> | <small>0.75M</small> | <small>37 s</small> | <small>$\color{#53803A}{0.9339 ± 0.0004}$</small> | <small>$\color{#51803B}{0.9305 ± 0.0019}$</small> | <small>$\color{#1E8449}{0.8317}$</small> | <small>$\color{#1E8449}{0.755}$</small> | <small>$\color{#308344}{0.9489}$</small> | <small>$\color{#1E8449}{0.9379}$</small> | <small>$\color{#1E8449}{0.9350}$</small> | <small>$\color{#1E8449}{0.9274}$</small> | <small>$\color{#B87420}{0.9031}$</small> |
| | <small>**ResNet, large**</small> | <small>5.5M</small> | <small>96 s</small> | <small>$\color{#1E8449}{0.9345 ± 0.0007}$</small> | <small>$\color{#1E8449}{0.9313 ± 0.0004}$</small> | <small>$\color{#238448}{0.8314}$</small> | <small>$\color{#777E31}{0.753}$</small> | <small>$\color{#1E8449}{0.9491}$</small> | <small>$\color{#308344}{0.9375}$</small> | <small>$\color{#4E813C}{0.9345}$</small> | <small>$\color{#1E8449}{0.9274}$</small> | <small>$\color{#1E8449}{0.9079}$</small> |
| <small>BiLSTM</small> | <small>small (hidden 128)</small> | <small>0.64M</small> | <small>213 s</small> | <small>$\color{#C0392B}{0.9310 ± 0.0006}$</small> | <small>$\color{#B96D21}{0.9284 ± 0.0009}$</small> | <small>$\color{#BA6722}{0.8229}$</small> | <small>$\color{#C0392B}{0.748}$</small> | <small>$\color{#B87520}{0.9474}$</small> | <small>$\color{#697F34}{0.9363}$</small> | <small>$\color{#B87520}{0.9333}$</small> | <small>$\color{#BF3E2A}{0.9199}$</small> | <small>$\color{#7B7D2F}{0.9052}$</small> |
| | <small>**large (hidden 256)**</small> | <small>2.3M</small> | <small>1001 s</small> | <small>$\color{#BE4728}{0.9314 ± 0.0008}$</small> | <small>$\color{#C0392B}{0.9264 ± 0.0009}$</small> | <small>$\color{#BD4D27}{0.8202}$</small> | <small>$\color{#777E31}{0.753}$</small> | <small>$\color{#C0392B}{0.9459}$</small> | <small>$\color{#C0392B}{0.9314}$</small> | <small>$\color{#B87121}{0.9332}$</small> | <small>$\color{#C0392B}{0.9196}$</small> | <small>$\color{#BA6722}{0.9022}$</small> |
| <small>CNN + Transformer</small> | <small>small (d 128, 2 layers)</small> | <small>0.80M</small> | <small>49 s</small> | <small>$\color{#BE4429}{0.9313 ± 0.0003}$</small> | <small>$\color{#BE4429}{0.9268 ± 0.0013}$</small> | <small>$\color{#C0392B}{0.8180}$</small> | <small>$\color{#C0392B}{0.748}$</small> | <small>$\color{#BA6523}{0.9470}$</small> | <small>$\color{#BF4129}{0.9318}$</small> | <small>$\color{#BD4D27}{0.9323}$</small> | <small>$\color{#AF7A21}{0.9237}$</small> | <small>$\color{#C0392B}{0.8990}$</small> |
| | <small>**large (d 256, 4 layers)**</small> | <small>4.7M</small> | <small>86 s</small> | <small>$\color{#B96822}{0.9323 ± 0.0010}$</small> | <small>$\color{#AE7A21}{0.9290 ± 0.0011}$</small> | <small>$\color{#BC5226}{0.8207}$</small> | <small>$\color{#A17B25}{0.752}$</small> | <small>$\color{#9A7B27}{0.9478}$</small> | <small>$\color{#B87320}{0.9343}$</small> | <small>$\color{#C0392B}{0.9318}$</small> | <small>$\color{#AF7A21}{0.9237}$</small> | <small>$\color{#348243}{0.9073}$</small> |

(NORM … HYP: per-class test AUROC. The ResNet runs are identical to `base` and `w256d2dp` of S2.)

- **Convolution fits ECG best.** The selected ResNet reaches 0.9313 ± 0.0004. That is 0.0023 above the CNN + Transformer (0.9290 ± 0.0011) and 0.0049 above the BiLSTM (0.9264 ± 0.0009).
  - The ranking is the same on the validation set (0.9345, 0.9323, 0.9314).
  - The ResNet is higher than both other families on every seed (test 0.9314, 0.9316, 0.9308, compared with at most 0.9300).
  - Even the small ResNet (0.75M parameters, 37 s) is better on average than every model of the other two families.

- **The advantage of convolution is largest for MI and CD.** Compared with the other two selected models, the ResNet is ahead for:
  - MI by 0.003–0.006;
  - CD by 0.004–0.008;
  - NORM and STTC by 0.001–0.003.

  For HYP it is level with the CNN + Transformer and 0.006 above the BiLSTM. A plausible explanation: myocardial infarction and conduction disturbances show in local waveform shape (Q waves, ST segments, QRS width), which local filters with a growing receptive field capture directly.

- **Attention and recurrence are close; neither matches convolution.** The selected CNN + Transformer is above the selected BiLSTM on validation, on test (+0.0026) and on every seed. However, the small BiLSTM, not selected because its validation score is 0.0004 lower, reaches 0.9284 on test. That is nearly level with the CNN + Transformer, so the order of these two families is not robust; the gap to convolution is. The CNN + Transformer starts with the same stem and residual blocks as the ResNet, then uses attention instead of further convolutions. That choice costs 0.0023 macro AUROC, so relating all positions of the 2.5 s window to each other adds nothing the convolutions miss.

- **Model size matters little.** Going from the small to the large size changes test macro AUROC by +0.0008 for the ResNet, −0.0020 for the BiLSTM and +0.0022 for the CNN + Transformer. All of these are of the order of the seed-to-seed variation, except perhaps for the Transformer.

- **Numerical precision does not explain the BiLSTM result.** Under the bf16 autocast of the standard protocol, PyTorch runs cuDNN LSTMs in float16 rather than bf16, and the protocol has no loss scaling for float16. The BiLSTM therefore computes its recurrence in float32 (`fp32_lstm`), and the rest of the model stays in bf16. A first run with the float16 recurrence (kept as diagnostic runs with the tag `fp16lstm` in `runs/e2_operation/`) reached the same validation macro AUROC: 0.9308 and 0.9312, compared with 0.9310 and 0.9314. Its test values were 0.9277 and 0.9278. The float32 recurrence is 3.5–4 times slower.

- **The sequence models overfit more towards the end of training.** Under the same 50-epoch schedule, validation macro AUROC falls from its best epoch to the last epoch by 0.006 for the large BiLSTM and 0.005 for the large CNN + Transformer, but only 0.002–0.003 for the ResNets. Their validation loss also rises more:

  | <small>Model</small> | <small>Best validation loss</small> | <small>At epoch 50</small> |
  |:------|--------------------:|------------:|
  | <small>BiLSTM, large</small> | <small>0.29</small> | <small>0.34</small> |
  | <small>CNN + Transformer, large</small> | <small>0.30</small> | <small>0.35</small> |
  | <small>ResNet, large</small> | <small>0.28</small> | <small>0.31</small> |

  The best-validation checkpoint absorbs this. Still, the augmentation selected for the ResNet in E3 may not be the best one for the other two families; E3 is to be repeated for them.

- **The BiLSTM is the most expensive model.** The recurrence processes the 250 time steps one after another. The large BiLSTM therefore needs 1001 s for 50 epochs, ten times the large ResNet. Even with the faster float16 recurrence it needed 290 s, three times the large ResNet.

- **Agreement with the benchmark.** The PTB-XL benchmark also ranks its convolutional `resnet1d_wang` (0.930) above its recurrent models (LSTM 0.927, bidirectional LSTM 0.921) [[4]](#references). The BiLSTMs trained here with the standard protocol reach 0.926–0.928.

Convolution is therefore kept as the operation type for the remaining experiments. Its best configuration, the ResNet with the lead-wise stem from S2, reaches 0.9328.

<p align="center">
  <img src="runs/e2_operation/figures/curves.png" alt="E2 learning curves" width="100%">
</p>

</details>

<details>
<summary><a id="e3-which-augmentations-help-none-vs-noise-baseline-wander-scaling-time-shift-random-crop"></a><h3 style="display:inline">E3: Which augmentations help? None vs noise, baseline wander, scaling, time shift, random crop</h3></summary>

```bash
python -m src.experiments.e3_augment                      # ResNet1D-10
python -m src.experiments.e3_augment --models <model>     # further models, same folder
```

ResNet1D-10, the reference model selected in E1, is trained with the standard protocol under seven conditions: no augmentation, each of the five augmentations alone, and all five combined. Noise, baseline wander, scaling and time shift are applied independently to each training record with probability 0.5. Random crop is applied to every training record, since all inputs of a batch must have the same length, and always after the other augmentations. Validation and test data are never augmented. Amplitudes are given in units of the per-lead standard deviation, since augmentation is applied after standardization.

| <small>Augmentation</small> | <small>Definition</small> |
|:-------------|:-----------|
| <small>**Noise**</small> | <small>additive Gaussian noise, σ ~ U(0.05, 0.15) (SNR ≈ 16–26 dB)</small> |
| <small>**Baseline wander**</small> | <small>per-lead sinusoid, 0.05–0.5 Hz, amplitude ~ U(0.1, 0.5), random phase</small> |
| <small>**Scaling**</small> | <small>one amplitude factor per record, ~ U(0.8, 1.2)</small> |
| <small>**Time shift**</small> | <small>circular shift of all leads by up to ± 1 s</small> |
| <small>**Random crop**</small> | <small>random 2.5 s window of each record (the crop length of the benchmark [[4]](#references))</small> |
| <small>**All combined**</small> | <small>the five above: the first four each with its own probability 0.5, followed by the crop</small> |

A model trained on random crops only ever sees 2.5 s inputs, so it is also validated and tested on 2.5 s inputs: each 10 s record is divided into 7 overlapping windows (stride 1.25 s), and the record's probability for each class is the maximum over the 7 windows, as in the benchmark [[4]](#references). This applies to the *random crop* and *all combined* conditions; all other models are evaluated on the full 10 s record.

The strengths were fixed by visual inspection of augmented records (clearly visible but physiologically plausible), not by validation or test performance. The same seven conditions will be repeated for every model added in E2.

#### Results

*ResNet1D-10, test set (fold 10), mean ± std over 3 seeds; Δ relative to no augmentation:*

| <small>Condition</small> | <small>Macro AUROC</small> | <small>Δ</small> | <small>Macro AUPRC</small> | <small>Macro F1 @0.5</small> | <small>Macro F1 @tuned</small> | <small>NORM</small> | <small>MI</small> | <small>STTC</small> | <small>CD</small> | <small>HYP</small> |
|:----------|------------:|--:|------------:|--------------:|----------------:|-----:|---:|-----:|---:|----:|
| <small>None</small> | <small>$\color{#BF3E2A}{0.9121 ± 0.0007}$</small> | <small>—</small> | <small>$\color{#BF3F2A}{0.7960 ± 0.0016}$</small> | <small>$\color{#BB5C25}{0.6996 ± 0.0231}$</small> | <small>$\color{#BF3F2A}{0.7232 ± 0.0045}$</small> | <small>$\color{#BE4828}{0.9350}$</small> | <small>$\color{#BF4229}{0.9020}$</small> | <small>$\color{#C0392B}{0.9270}$</small> | <small>$\color{#C0392B}{0.9010}$</small> | <small>$\color{#BF3D2A}{0.8952}$</small> |
| <small>Noise</small> | <small>$\color{#BE4429}{0.9130 ± 0.0007}$</small> | <small>+0.0009</small> | <small>$\color{#BE4A28}{0.7982 ± 0.0010}$</small> | <small>$\color{#B96822}{0.7053 ± 0.0223}$</small> | <small>$\color{#BC5526}{0.7278 ± 0.0032}$</small> | <small>$\color{#BD4C27}{0.9354}$</small> | <small>$\color{#BE4529}{0.9030}$</small> | <small>$\color{#BE4A28}{0.9281}$</small> | <small>$\color{#BE4429}{0.9032}$</small> | <small>$\color{#BF4129}{0.8955}$</small> |
| <small>Baseline wander</small> | <small>$\color{#BE4429}{0.9130 ± 0.0011}$</small> | <small>+0.0009</small> | <small>$\color{#BE4728}{0.7976 ± 0.0022}$</small> | <small>$\color{#BB5C25}{0.6991 ± 0.0283}$</small> | <small>$\color{#BC5326}{0.7277 ± 0.0069}$</small> | <small>$\color{#BE4A28}{0.9352}$</small> | <small>$\color{#BE4A28}{0.9041}$</small> | <small>$\color{#BD4D27}{0.9284}$</small> | <small>$\color{#BF3F2A}{0.9021}$</small> | <small>$\color{#BF3F2A}{0.8954}$</small> |
| <small>Scaling</small> | <small>$\color{#C0392B}{0.9115 ± 0.0008}$</small> | <small>−0.0006</small> | <small>$\color{#C0392B}{0.7945 ± 0.0020}$</small> | <small>$\color{#C0392B}{0.6825 ± 0.0159}$</small> | <small>$\color{#C0392B}{0.7219 ± 0.0017}$</small> | <small>$\color{#C0392B}{0.9337}$</small> | <small>$\color{#C0392B}{0.8997}$</small> | <small>$\color{#C03C2A}{0.9272}$</small> | <small>$\color{#BF3E2A}{0.9019}$</small> | <small>$\color{#C0392B}{0.8948}$</small> |
| <small>Time shift</small> | <small>$\color{#B27920}{0.9201 ± 0.0021}$</small> | <small>+0.0080</small> | <small>$\color{#AE7A21}{0.8097 ± 0.0042}$</small> | <small>$\color{#A57A24}{0.7178 ± 0.0032}$</small> | <small>$\color{#9C7B26}{0.7380 ± 0.0051}$</small> | <small>$\color{#817D2E}{0.9413}$</small> | <small>$\color{#B87021}{0.9140}$</small> | <small>$\color{#BC5625}{0.9290}$</small> | <small>$\color{#B87121}{0.9116}$</small> | <small>$\color{#598039}{0.9046}$</small> |
| <small>**Random crop**</small> | <small>$\color{#1E8449}{0.9280 ± 0.0015}$</small> | <small>**+0.0159**</small> | <small>$\color{#2E8345}{0.8216 ± 0.0047}$</small> | <small>$\color{#1E8449}{0.7453 ± 0.0028}$</small> | <small>$\color{#1E8449}{0.7491 ± 0.0006}$</small> | <small>$\color{#298346}{0.9445}$</small> | <small>$\color{#298346}{0.9321}$</small> | <small>$\color{#49813D}{0.9343}$</small> | <small>$\color{#44813F}{0.9223}$</small> | <small>$\color{#1E8449}{0.9069}$</small> |
| <small>**All combined**</small> | <small>$\color{#208448}{0.9279 ± 0.0010}$</small> | <small>**+0.0158**</small> | <small>$\color{#1E8449}{0.8232 ± 0.0057}$</small> | <small>$\color{#298346}{0.7430 ± 0.0027}$</small> | <small>$\color{#3B8241}{0.7465 ± 0.0035}$</small> | <small>$\color{#1E8449}{0.9449}$</small> | <small>$\color{#1E8449}{0.9332}$</small> | <small>$\color{#1E8449}{0.9355}$</small> | <small>$\color{#1E8449}{0.9252}$</small> | <small>$\color{#B77620}{0.9006}$</small> |

(NORM … HYP: per-class test AUROC. The *none* runs are identical to the ResNet1D-10 runs of E1, as training is deterministic for a given seed.)

- **Random crop provides by far the largest benefit.** It raises macro AUROC from 0.9121 to 0.9280 (+0.0159), twice the improvement obtained with time shift; all three random-crop runs (0.9265–0.9294) exceed all three time-shift runs (0.9177–0.9215), which in turn exceed all three runs without augmentation (0.9113–0.9126). The improvement is consistent across metrics (macro AUPRC +0.026, macro F1 at validation-tuned thresholds +0.026) and across all five superclasses, with the largest gains for MI (+0.030) and CD (+0.021) and the smallest for STTC (+0.007).

- **Random crop largely prevents overfitting.** The best validation epoch moves from epoch 10 to epoch 33. At the end of training the training loss is 0.16 compared with 0.0006 without augmentation, the validation loss is 0.33 compared with 0.77, and the validation macro AUROC at epoch 50 (0.928) remains close to its maximum (0.931), whereas without augmentation it falls from 0.919 to 0.903. Since each epoch presents a different 2.5 s segment of every recording, the network can no longer memorize individual recordings, which was identified in E1 as the factor limiting generalization.

- **The random-crop condition combines two changes.** In addition to training on crops, these models are evaluated on 7 overlapping windows per record with the per-class maximum, which acts as a form of test-time augmentation. The present design does not separate the contribution of the training procedure from that of the windowed evaluation.

- **Time shift provides a clear but smaller benefit.** Time shift raises macro AUROC by 0.0080 and delays overfitting (best validation epoch 17; final validation loss 0.47) without preventing it: validation AUROC still declines after its maximum. A plausible explanation is that, although global average pooling makes the network approximately invariant to translation, strided downsampling breaks exact invariance, and the circular shift additionally changes which beats lie at the boundaries of the recording, so that each shifted record is effectively a new training sequence.

- **Noise, baseline wander and scaling have no measurable effect.** Their differences from the condition without augmentation (+0.0009, +0.0009 and −0.0006) are within the seed-to-seed variation, and none of them changes the course of overfitting. In these experiments, augmentations that perturb the amplitude of the signal do not slow the memorization of recordings, whereas those that change its temporal content do.

- **Combining all augmentations provides no benefit beyond random crop.** The combined condition (0.9279 ± 0.0010) is indistinguishable from random crop alone (0.9280 ± 0.0015); the only per-class difference of note is a lower HYP AUROC (0.9006 compared with 0.9069). The improvement is therefore attributable to random crop.

- **Relation to the benchmark.** With random crop, ResNet1D-10 reaches a macro AUROC of 0.9280, within 0.002 of the 0.930 reported for `resnet1d_wang` [[4]](#references) and within the 95% bootstrap confidence interval of each random-crop run (upper bounds 0.933–0.936). The gap identified in E1 is therefore explained by the training and evaluation procedure rather than by the architecture.

- **F1 at a fixed threshold of 0.5 is unstable without temporal augmentation.** Without augmentation and with noise, baseline wander or scaling, its standard deviation across seeds (0.016–0.028) is larger than most differences between these conditions, whereas with time shift or random crop it is approximately 0.003. Threshold-independent metrics (AUROC, AUPRC) and F1 at validation-tuned thresholds remain the more reliable basis for comparison.

<p align="center">
  <img src="runs/e3_augment/figures/curves.png" alt="E3 learning curves" width="100%">
</p>

</details>

<details>
<summary><a id="e4-how-many-leads-are-needed-12-leads-vs-6-limb-leads-vs-lead-ii-only"></a><h3 style="display:inline">E4: How many leads are needed? 12 leads vs 6 limb leads vs lead II only</h3></summary>

```bash
python -m src.experiments.e4_leads    # 3 lead sets x 3 seeds
```

The reference model ResNet1D-10 (E1) is trained with the standard protocol and random 2.5 s crops (E3) on three lead sets; only the input leads change. Lead subsets are selected when the data are loaded, so every lead set uses the same cache, records and split.

| <small>Lead set</small> | <small>Leads</small> |
|:---------|:------|
| <small>**12 leads**</small> | <small>I, II, III, aVR, aVL, aVF, V1–V6</small> |
| <small>**6 limb leads**</small> | <small>I, II, III, aVR, aVL, aVF</small> |
| <small>**Lead II**</small> | <small>II</small> |

#### Results

*Test set (fold 10), mean ± std over 3 seeds; 95% bootstrap CI of the macro AUROC averaged over seeds.*

| <small>Lead set</small> | <small>Val macro AUROC</small> | <small>Test macro AUROC</small> | <small>95% CI</small> | <small>Macro AUPRC</small> | <small>Macro F1 @tuned</small> | <small>NORM</small> | <small>MI</small> | <small>STTC</small> | <small>CD</small> | <small>HYP</small> |
|:---------|----------------:|-----------------:|:------:|------------:|----------------:|-----:|---:|-----:|---:|----:|
| <small>**12 leads**</small> | <small>$\color{#1E8449}{0.9314 ± 0.0010}$</small> | <small>$\color{#1E8449}{0.9280 ± 0.0015}$</small> | <small>[0.921, 0.935]</small> | <small>$\color{#1E8449}{0.8216}$</small> | <small>$\color{#1E8449}{0.7491}$</small> | <small>$\color{#1E8449}{0.9445}$</small> | <small>$\color{#1E8449}{0.9321}$</small> | <small>$\color{#1E8449}{0.9343}$</small> | <small>$\color{#1E8449}{0.9223}$</small> | <small>$\color{#1E8449}{0.9069}$</small> |
| <small>6 limb leads</small> | <small>$\color{#A77A23}{0.9032 ± 0.0007}$</small> | <small>$\color{#AF7A21}{0.8953 ± 0.0004}$</small> | <small>[0.888, 0.904]</small> | <small>$\color{#AC7A22}{0.7570}$</small> | <small>$\color{#B87420}{0.6913}$</small> | <small>$\color{#A17B25}{0.9252}$</small> | <small>$\color{#AA7A23}{0.8873}$</small> | <small>$\color{#827D2E}{0.9166}$</small> | <small>$\color{#BE4828}{0.8847}$</small> | <small>$\color{#A17B25}{0.8627}$</small> |
| <small>Lead II</small> | <small>$\color{#C0392B}{0.8681 ± 0.0002}$</small> | <small>$\color{#C0392B}{0.8591 ± 0.0002}$</small> | <small>[0.850, 0.868]</small> | <small>$\color{#C0392B}{0.6831}$</small> | <small>$\color{#C0392B}{0.6412}$</small> | <small>$\color{#C0392B}{0.8999}$</small> | <small>$\color{#C0392B}{0.8337}$</small> | <small>$\color{#C0392B}{0.8797}$</small> | <small>$\color{#C0392B}{0.8793}$</small> | <small>$\color{#C0392B}{0.8031}$</small> |

(NORM … HYP: per-class test AUROC. The 12-lead runs reproduce the random-crop runs of E3 exactly.)

- **All 12 leads are needed.** Without the six precordial leads, macro AUROC drops by 0.033; with lead II alone, by 0.069. The 95% confidence intervals of the three lead sets do not overlap. Macro AUPRC falls even more, from 0.822 to 0.757 and 0.683.

- **The precordial leads carry the information for MI, HYP and CD.** Without V1–V6, the largest losses are:

  | <small>Class</small> | <small>Drop</small> |
  |:------|-----:|
  | <small>MI</small> | <small>−0.045</small> |
  | <small>HYP</small> | <small>−0.044</small> |
  | <small>CD</small> | <small>−0.038</small> |
  | <small>NORM</small> | <small>−0.019</small> |
  | <small>STTC</small> | <small>−0.018</small> |

  This agrees with how these conditions are read: anterior infarctions appear mainly in V1–V4, the voltage criteria for left ventricular hypertrophy use V1, V5 and V6, and bundle branch blocks are recognized from the QRS shape in V1 and V6.

- **Lead II alone keeps most of the limb-lead information for CD, but not for MI and HYP.** Relative to the 6 limb leads, lead II alone loses only 0.005 for CD, but 0.054 for MI and 0.060 for HYP. A plausible explanation: conduction disturbances such as atrioventricular blocks show as timing changes, e.g. of the PR interval, that are visible in a single lead. Infarction and hypertrophy, by contrast, depend on how the signal is distributed across leads.

- **Fewer leads remove information, not just noise.** With fewer leads the model memorizes the training set less: at epoch 50 the training loss is 0.16, 0.25 and 0.32 for 12, 6 and 1 lead. The best validation epoch also comes later: 32–35, 28–38 and 45–48. Yet the minimum validation loss rises, from 0.29 to 0.33 and 0.37, so the missing leads cost accuracy rather than reducing overfitting.

- **The cost is unchanged.** Only the first convolution depends on the number of leads (2.80–2.81M parameters), and every run takes about 40 s.

The same comparison for the SVM baseline (B0) is planned.

<p align="center">
  <img src="runs/e4_leads/figures/curves.png" alt="E4 learning curves" width="100%">
</p>

</details>

<details>
<summary><a id="e5-does-finer-resolution-help-100-hz-vs-500-hz"></a><h3 style="display:inline">E5: Does finer resolution help? 100 Hz vs 500 Hz</h3></summary>

*Planned.* The 500 Hz signal cache is built once with `python -m src.data.preprocess --fs 500`; runs then use it with `--fs 500`.

</details>

---

## Supplementary study: Optimizing the ResNet architecture

S1 and S2 study the ResNet itself in more depth. S1 examines individual design choices; S2 searches the ResNet family for the best single model. The ResNet thus received considerably more design effort than the other model families: 8 design variants in S1 and 30 architecture variants in S2, compared with two sizes each for the BiLSTM and the CNN + Transformer in E2. For this reason the study is reported separately from the research questions.

<details>
<summary><a id="s1-which-resnet-design-choices-help-custom-resnet-variants"></a><h3 style="display:inline">S1: Which ResNet design choices help? Custom ResNet variants</h3></summary>

```bash
python -m src.experiments.s1_resnet_design
```

E1 compared classic ResNets that differ only in depth. S1 starts from the classic ResNet1D-10 and changes one design choice at a time, using three choices in which the classic layout differs from `resnet1d_wang` (B0); the result is a custom ResNet. All models are trained with the standard protocol and random crop, the best augmentation of E3, so that only the architecture changes.

| <small>Model</small> | <small>Change from the classic ResNet1D-10</small> | <small>Motivation</small> |
|:------|:------------------------------------|:-----------|
| <small>`resnet1d10`</small> | <small>none (classic layout)</small> | <small>baseline</small> |
| <small>`resnet1d10_stem1`</small> | <small>stem convolution with stride 1 and no max-pooling</small> | <small>4 times finer time axis: on a 2.5 s crop, 32 instead of 8 time steps remain after the last stage</small> |
| <small>`resnet1d10_chead`</small> | <small>head of the benchmark: concatenated max and average pooling, batch-norm, dropout 0.25 and 0.5, hidden layer of 128 units</small> | <small>max pooling retains short, localized findings; dropout regularizes</small> |
| <small>`resnet1d10_w128`</small> | <small>constant width of 128 channels instead of 64–128–256–512</small> | <small>a quarter of the parameters, less capacity to memorize</small> |
| <small>`resnet1d10_stem1_chead_w128`</small> | <small>all three</small> | |
| <small>`resnet1d18`, `resnet1d34`</small> | <small>deeper classic ResNets</small> | <small>does the E1 result change once random crop limits overfitting?</small> |
| <small>`resnet1d_wang`</small> | <small>benchmark architecture</small> | <small>fair comparison with B0 under the standard protocol</small> |

#### Results

*Standard protocol with random crop, test set (fold 10), mean ± std over 3 seeds; Δ relative to the classic ResNet1D-10:*

| <small>Model</small> | <small>Params</small> | <small>Train time</small> | <small>Macro AUROC</small> | <small>Δ</small> | <small>Macro AUPRC</small> | <small>Macro F1 @0.5</small> | <small>Macro F1 @tuned</small> | <small>NORM</small> | <small>MI</small> | <small>STTC</small> | <small>CD</small> | <small>HYP</small> |
|:------|-------:|-----------:|------------:|--:|------------:|--------------:|----------------:|-----:|---:|-----:|---:|----:|
| <small>ResNet1D-10 (classic)</small> | <small>2.8M</small> | <small>31 s</small> | <small>$\color{#B87320}{0.9280 ± 0.0015}$</small> | <small>—</small> | <small>$\color{#BA6223}{0.8216 ± 0.0047}$</small> | <small>$\color{#AF7A21}{0.7453 ± 0.0028}$</small> | <small>$\color{#B57920}{0.7491 ± 0.0006}$</small> | <small>$\color{#C0392B}{0.9445}$</small> | <small>$\color{#BA6323}{0.9321}$</small> | <small>$\color{#46813E}{0.9343}$</small> | <small>$\color{#BA6223}{0.9223}$</small> | <small>$\color{#1E8449}{0.9069}$</small> |
| <small>+ stride-1 stem</small> | <small>2.8M</small> | <small>36 s</small> | <small>$\color{#BA6623}{0.9275 ± 0.0020}$</small> | <small>−0.0005</small> | <small>$\color{#B87021}{0.8232 ± 0.0053}$</small> | <small>$\color{#C0392B}{0.7361 ± 0.0059}$</small> | <small>$\color{#B96A22}{0.7474 ± 0.0086}$</small> | <small>$\color{#8D7C2B}{0.9473}$</small> | <small>$\color{#AF7A21}{0.9338}$</small> | <small>$\color{#C0392B}{0.9295}$</small> | <small>$\color{#BA6223}{0.9223}$</small> | <small>$\color{#827D2E}{0.9046}$</small> |
| <small>+ concat-pooling head</small> | <small>2.9M</small> | <small>35 s</small> | <small>$\color{#BB5D24}{0.9272 ± 0.0019}$</small> | <small>−0.0008</small> | <small>$\color{#B96B22}{0.8228 ± 0.0063}$</small> | <small>$\color{#B96D21}{0.7431 ± 0.0043}$</small> | <small>$\color{#BA6623}{0.7471 ± 0.0052}$</small> | <small>$\color{#BB5F24}{0.9458}$</small> | <small>$\color{#C0392B}{0.9293}$</small> | <small>$\color{#727E32}{0.9335}$</small> | <small>$\color{#BA6523}{0.9224}$</small> | <small>$\color{#797D30}{0.9048}$</small> |
| <small>+ width 128</small> | <small>0.72M</small> | <small>31 s</small> | <small>$\color{#BA6623}{0.9275 ± 0.0015}$</small> | <small>−0.0005</small> | <small>$\color{#BB5A25}{0.8207 ± 0.0010}$</small> | <small>$\color{#B96E21}{0.7433 ± 0.0040}$</small> | <small>$\color{#B96D21}{0.7477 ± 0.0029}$</small> | <small>$\color{#BC5326}{0.9454}$</small> | <small>$\color{#BB5A25}{0.9315}$</small> | <small>$\color{#777E31}{0.9334}$</small> | <small>$\color{#BE4728}{0.9207}$</small> | <small>$\color{#348243}{0.9064}$</small> |
| <small>**+ all three**</small> | <small>0.75M</small> | <small>38 s</small> | <small>$\color{#1E8449}{0.9305 ± 0.0019}$</small> | <small>**+0.0025**</small> | <small>$\color{#1E8449}{0.8317 ± 0.0051}$</small> | <small>$\color{#1E8449}{0.7536 ± 0.0045}$</small> | <small>$\color{#1E8449}{0.7554 ± 0.0048}$</small> | <small>$\color{#1E8449}{0.9489}$</small> | <small>$\color{#1E8449}{0.9379}$</small> | <small>$\color{#1E8449}{0.9350}$</small> | <small>$\color{#1E8449}{0.9274}$</small> | <small>$\color{#B87320}{0.9031}$</small> |
| <small>ResNet1D-18 (classic)</small> | <small>6.3M</small> | <small>46 s</small> | <small>$\color{#BE4A28}{0.9265 ± 0.0007}$</small> | <small>−0.0015</small> | <small>$\color{#BF4129}{0.8179 ± 0.0011}$</small> | <small>$\color{#BB5F24}{0.7413 ± 0.0009}$</small> | <small>$\color{#C0392B}{0.7426 ± 0.0024}$</small> | <small>$\color{#BE4429}{0.9449}$</small> | <small>$\color{#BB5C25}{0.9316}$</small> | <small>$\color{#B77620}{0.9321}$</small> | <small>$\color{#C0392B}{0.9199}$</small> | <small>$\color{#947B29}{0.9042}$</small> |
| <small>ResNet1D-34 (classic)</small> | <small>11.9M</small> | <small>78 s</small> | <small>$\color{#C0392B}{0.9259 ± 0.0007}$</small> | <small>−0.0021</small> | <small>$\color{#C0392B}{0.8170 ± 0.0032}$</small> | <small>$\color{#B7781F}{0.7447 ± 0.0017}$</small> | <small>$\color{#BB5A25}{0.7459 ± 0.0013}$</small> | <small>$\color{#BC5625}{0.9455}$</small> | <small>$\color{#BD4C27}{0.9306}$</small> | <small>$\color{#BB5A25}{0.9309}$</small> | <small>$\color{#BA6523}{0.9224}$</small> | <small>$\color{#C0392B}{0.8999}$</small> |
| <small>`resnet1d_wang`</small> | <small>0.47M</small> | <small>32 s</small> | <small>$\color{#897C2C}{0.9289 ± 0.0006}$</small> | <small>+0.0009</small> | <small>$\color{#867D2C}{0.8267 ± 0.0020}$</small> | <small>$\color{#A57A24}{0.7459 ± 0.0072}$</small> | <small>$\color{#967B28}{0.7504 ± 0.0060}$</small> | <small>$\color{#AF7A21}{0.9468}$</small> | <small>$\color{#AA7A23}{0.9340}$</small> | <small>$\color{#947B29}{0.9329}$</small> | <small>$\color{#897C2C}{0.9248}$</small> | <small>$\color{#4E813C}{0.9058}$</small> |

(NORM … HYP: per-class test AUROC. The classic ResNet1D-10 runs are identical to the random-crop runs of E3.)

- **No single design change improves on the classic ResNet1D-10.** The stride-1 stem (−0.0005), the concatenated-pooling head (−0.0008) and the constant width of 128 channels (−0.0005) each leave macro AUROC unchanged within the seed-to-seed variation. The constant width nevertheless achieves equivalent performance with a quarter of the parameters (0.72M compared with 2.8M).

- **The three changes improve performance in combination.** The variant combining all three reaches 0.9305 ± 0.0019 (+0.0025). It exceeds the classic ResNet1D-10 on every seed (by 0.0021, 0.0035 and 0.0018) and has the highest validation macro AUROC of all eight models on every seed (0.9336–0.9344), so the improvement is consistent, although small relative to the seed-to-seed variation. It is also visible in macro AUPRC (+0.010) and in macro F1 at validation-tuned thresholds (+0.006), and is concentrated in MI (+0.006), CD (+0.005) and NORM (+0.004), whereas HYP is slightly lower (−0.004).

- **The changes interact.** On its own, the finer time axis of the stride-1 stem makes the network overfit sooner: the best validation epoch moves from 33 to 22, the final training loss is lower (0.11 compared with 0.16), the final validation loss higher (0.38 compared with 0.33), and the validation AUROC at epoch 50 lower (0.923 compared with 0.928). Combined with the regularizing head and the reduced width, the finer resolution no longer leads to earlier overfitting (best epoch 32, final validation loss 0.31, validation AUROC at epoch 50 of 0.932). A plausible explanation is that the finer time axis can only be exploited once the capacity of the network to memorize the training set is limited.

- **Depth still does not help, even with random crop.** ResNet1D-18 (0.9265) and ResNet1D-34 (0.9259) do not improve on ResNet1D-10 (0.9280) and overfit increasingly with depth: the final validation loss rises from 0.33 to 0.36 and 0.40, and the validation AUROC at epoch 50 falls from 0.928 to 0.924 and 0.920. The conclusion of E1 therefore holds once memorization is limited by random crop.

- **Fair comparison with `resnet1d_wang`.** Under the standard protocol with random crop, `resnet1d_wang` reaches 0.9289 ± 0.0006, equivalent to the classic ResNet1D-10 (+0.0009) and below the custom ResNet combining all three changes (−0.0016), which is higher on two of the three seeds and equal on the third (0.9315 vs 0.9291, 0.9317 vs 0.9293, 0.9283 vs 0.9282). Trained with its own procedure in B0, `resnet1d_wang` reached 0.9324 with the same checkpoint selection; the difference of 0.0035 is therefore attributable to the training procedure (learning rate, Adam β₂, standardization) and not to the architecture. Under a fixed training protocol, the custom ResNet matches and slightly exceeds `resnet1d_wang`.

<p align="center">
  <img src="runs/s1_resnet_design/figures/curves.png" alt="S1 learning curves" width="100%">
</p>

</details>

<details>
<summary><a id="s2-how-far-can-a-resnet-go-architecture-search-within-the-resnet-family"></a><h3 style="display:inline">S2: How far can a ResNet go? Architecture search within the ResNet family</h3></summary>

```bash
python -m src.experiments.s2_resnet_search                                # all variants, seeds 0-2
python -m src.experiments.s2_resnet_search --variants base leadwise --seeds 0
python -m src.experiments.s2_resnet_search --report-only --show-test       # final report with test set
```

Starting from the best custom ResNet of S1 (`base`: stride-1 stem, concatenated-pooling head, constant width 128), S2 searches the ResNet family for the best single model. Only the architecture changes. Every variant is one ResNet, trained with the standard protocol and random crop as in S1, and evaluated exactly as in E3 and S1 (maximum over the 7 windows). There are no ensembles and no changes to training or inference. All variants are built with `src/models/resnet1d_plus.py`, whose default is identical to `base`.

**Selection.** Because 30 variants are compared, selection uses the validation set (fold 9) only. Every variant is trained on three seeds, and the variant with the highest mean validation macro AUROC is selected. The test set was evaluated only after this choice had been fixed, so its values below played no part in the selection.

The search ran in three rounds:
1. common ResNet modifications, one at a time and in combinations;
2. receptive field, downsampling, stem, block type and head;
3. combinations built on the only consistent improvement of round 2, the lead-wise stem.

*Sorted by validation macro AUROC; mean ± std over 3 seeds. Base receptive field: 99 samples (1.0 s).*

| <small>Variant</small> | <small>Change from `base`</small> | <small>Params</small> | <small>Val macro AUROC</small> | <small>Test macro AUROC</small> |
|:--------|:-------------------|-------:|----------------:|-----------------:|
| <small>**`lww256d2dp`** (selected)</small> | <small>lead-wise stem + width 256 + 2 blocks per stage + stochastic depth</small> | <small>5.5M</small> | <small>$\color{#1E8449}{0.9356 ± 0.0013}$</small> | <small>$\color{#4F803C}{0.9328 ± 0.0006}$</small> |
| <small>`lw16`</small> | <small>lead-wise stem with 16 filters per lead</small> | <small>0.77M</small> | <small>$\color{#3B8241}{0.9352 ± 0.0009}$</small> | <small>$\color{#1E8449}{0.9337 ± 0.0008}$</small> |
| <small>`leadwise`</small> | <small>lead-wise stem: grouped convolution (kernel 7) with 8 filters per lead, then a 1×1 convolution across the leads</small> | <small>0.76M</small> | <small>$\color{#51803B}{0.9349 ± 0.0004}$</small> | <small>$\color{#A17B25}{0.9313 ± 0.0016}$</small> |
| <small>`lwse`</small> | <small>lead-wise stem + squeeze-excitation</small> | <small>0.76M</small> | <small>$\color{#677F35}{0.9346 ± 0.0009}$</small> | <small>$\color{#B7771F}{0.9308 ± 0.0009}$</small> |
| <small>`lwsed2dp`</small> | <small>lead-wise stem + squeeze-excitation + 2 blocks + stochastic depth</small> | <small>1.4M</small> | <small>$\color{#677F35}{0.9346 ± 0.0004}$</small> | <small>$\color{#7B7D2F}{0.9320 ± 0.0005}$</small> |
| <small>`w256d2dp`</small> | <small>width 256 + 2 blocks + stochastic depth</small> | <small>5.5M</small> | <small>$\color{#6C7E34}{0.9345 ± 0.0007}$</small> | <small>$\color{#A17B25}{0.9313 ± 0.0004}$</small> |
| <small>`base`</small> | <small>—</small> | <small>0.75M</small> | <small>$\color{#967B28}{0.9339 ± 0.0004}$</small> | <small>$\color{#B87021}{0.9305 ± 0.0019}$</small> |
| <small>`sed2dp`</small> | <small>squeeze-excitation + 2 blocks + stochastic depth</small> | <small>1.4M</small> | <small>$\color{#967B28}{0.9339 ± 0.0002}$</small> | <small>$\color{#B27920}{0.9310 ± 0.0007}$</small> |
| <small>`k53`</small> | <small>kernel 5 in the first and 3 in the second convolution of each block (as `resnet1d_wang`)</small> | <small>0.62M</small> | <small>$\color{#9F7B26}{0.9338 ± 0.0011}$</small> | <small>$\color{#BA6523}{0.9300 ± 0.0022}$</small> |
| <small>`se`</small> | <small>squeeze-excitation on every residual branch [[12]](#references)</small> | <small>0.76M</small> | <small>$\color{#A57A24}{0.9337 ± 0.0009}$</small> | <small>$\color{#B87320}{0.9306 ± 0.0009}$</small> |
| <small>`d2dp`</small> | <small>`[2,2,2,2]` blocks, residual branches dropped with probability up to 0.2 [[13]](#references)</small> | <small>1.4M</small> | <small>$\color{#A57A24}{0.9337 ± 0.0002}$</small> | <small>$\color{#B7791F}{0.9309 ± 0.0014}$</small> |
| <small>`w256`</small> | <small>256 channels in every stage</small> | <small>2.9M</small> | <small>$\color{#AC7A22}{0.9336 ± 0.0009}$</small> | <small>$\color{#B87320}{0.9306 ± 0.0016}$</small> |
| <small>`blur`</small> | <small>anti-aliased downsampling: stride-1 convolution followed by blur pooling, also in the shortcut [[15]](#references)</small> | <small>0.75M</small> | <small>$\color{#B47920}{0.9335 ± 0.0013}$</small> | <small>$\color{#BC5526}{0.9293 ± 0.0008}$</small> |
| <small>`bneck50`</small> | <small>bottleneck blocks `[3,4,6,3]`, widths 256–2048 (1D ResNet-50) [[8]](#references)</small> | <small>19.0M</small> | <small>$\color{#B47920}{0.9335 ± 0.0011}$</small> | <small>$\color{#BC5625}{0.9294 ± 0.0006}$</small> |
| <small>`preact`</small> | <small>pre-activation blocks (BN–ReLU–conv) [[11]](#references)</small> | <small>0.75M</small> | <small>$\color{#B7771F}{0.9334 ± 0.0002}$</small> | <small>$\color{#BD4D27}{0.9290 ± 0.0019}$</small> |
| <small>`d2`</small> | <small>`[2,2,2,2]` blocks</small> | <small>1.4M</small> | <small>$\color{#B7771F}{0.9334 ± 0.0002}$</small> | <small>$\color{#BA6523}{0.9300 ± 0.0010}$</small> |
| <small>`ps25`</small> | <small>head dropout 0.25 instead of 0.5</small> | <small>0.75M</small> | <small>$\color{#B7771F}{0.9334 ± 0.0007}$</small> | <small>$\color{#B96A22}{0.9302 ± 0.0011}$</small> |
| <small>`s3w256`</small> | <small>3 stages + width 256</small> | <small>2.2M</small> | <small>$\color{#B87420}{0.9333 ± 0.0002}$</small> | <small>$\color{#BC5625}{0.9294 ± 0.0004}$</small> |
| <small>`k7`</small> | <small>kernel size 7 (receptive field 1.5 s)</small> | <small>1.0M</small> | <small>$\color{#B87420}{0.9333 ± 0.0004}$</small> | <small>$\color{#B87320}{0.9306 ± 0.0017}$</small> |
| <small>`s3`</small> | <small>one stage fewer (`[1,1,1]`)</small> | <small>0.57M</small> | <small>$\color{#B87121}{0.9332 ± 0.0008}$</small> | <small>$\color{#BF4229}{0.9285 ± 0.0017}$</small> |
| <small>`resd`</small> | <small>average pooling before the strided 1×1 shortcut (ResNet-D) [[14]](#references)</small> | <small>0.75M</small> | <small>$\color{#B96E21}{0.9331 ± 0.0002}$</small> | <small>$\color{#BD5027}{0.9291 ± 0.0009}$</small> |
| <small>`ps75`</small> | <small>head dropout 0.75 instead of 0.5</small> | <small>0.75M</small> | <small>$\color{#B96E21}{0.9331 ± 0.0005}$</small> | <small>$\color{#BC5226}{0.9292 ± 0.0004}$</small> |
| <small>`dilrf`</small> | <small>dilation 2 in stages 2–4 (receptive field 1.8 s) [[16]](#references)</small> | <small>0.75M</small> | <small>$\color{#B96B22}{0.9330 ± 0.0008}$</small> | <small>$\color{#BC5526}{0.9293 ± 0.0003}$</small> |
| <small>`deepstem`</small> | <small>three stem convolutions instead of one [[14]](#references)</small> | <small>0.81M</small> | <small>$\color{#BA6323}{0.9327 ± 0.0003}$</small> | <small>$\color{#BB5C25}{0.9296 ± 0.0004}$</small> |
| <small>`k3`</small> | <small>kernel size 3</small> | <small>0.49M</small> | <small>$\color{#BB6024}{0.9326 ± 0.0005}$</small> | <small>$\color{#B96A22}{0.9302 ± 0.0014}$</small> |
| <small>`w64`</small> | <small>64 channels in every stage</small> | <small>0.20M</small> | <small>$\color{#BB5A25}{0.9324 ± 0.0009}$</small> | <small>$\color{#BA6223}{0.9299 ± 0.0012}$</small> |
| <small>`stemk15`</small> | <small>stem kernel size 15 instead of 7</small> | <small>0.76M</small> | <small>$\color{#BB5A25}{0.9324 ± 0.0007}$</small> | <small>$\color{#BF3E2A}{0.9283 ± 0.0005}$</small> |
| <small>`k9`</small> | <small>kernel size 9 (receptive field 1.9 s)</small> | <small>1.3M</small> | <small>$\color{#BE4A28}{0.9319 ± 0.0009}$</small> | <small>$\color{#BC5825}{0.9295 ± 0.0001}$</small> |
| <small>`silu`</small> | <small>SiLU instead of ReLU</small> | <small>0.75M</small> | <small>$\color{#C03C2A}{0.9314 ± 0.0005}$</small> | <small>$\color{#BD4B28}{0.9289 ± 0.0012}$</small> |
| <small>`dil`</small> | <small>no downsampling, dilations 1/2/4/8 (dilated ResNet; receptive field 1.3 s) [[16]](#references)</small> | <small>0.70M</small> | <small>$\color{#C0392B}{0.9313 ± 0.0003}$</small> | <small>$\color{#C0392B}{0.9281 ± 0.0006}$</small> |

#### Results

*Selected model compared with `base` and the lead-wise stem alone; test set (fold 10), mean ± std over 3 seeds.*

| <small>Model</small> | <small>Params</small> | <small>Val macro AUROC</small> | <small>Test macro AUROC</small> | <small>Test macro AUPRC</small> | <small>Test macro F1 (tuned)</small> |
|:------|-------:|----------------:|-----------------:|-----------------:|----------------------:|
| <small>`base` (S1)</small> | <small>0.75M</small> | <small>$\color{#C0392B}{0.9339 ± 0.0004}$</small> | <small>$\color{#C0392B}{0.9305 ± 0.0019}$</small> | <small>$\color{#C0392B}{0.8317 ± 0.0051}$</small> | <small>$\color{#C0392B}{0.755 ± 0.005}$</small> |
| <small>`leadwise`</small> | <small>0.76M</small> | <small>$\color{#9C7B26}{0.9349 ± 0.0004}$</small> | <small>$\color{#BA6623}{0.9313 ± 0.0016}$</small> | <small>$\color{#BA6523}{0.8333 ± 0.0045}$</small> | <small>$\color{#C0392B}{0.755 ± 0.001}$</small> |
| <small>**`lww256d2dp`** (selected)</small> | <small>5.5M</small> | <small>$\color{#1E8449}{0.9356 ± 0.0013}$</small> | <small>$\color{#1E8449}{0.9328 ± 0.0006}$</small> | <small>$\color{#1E8449}{0.8365 ± 0.0025}$</small> | <small>$\color{#1E8449}{0.759 ± 0.006}$</small> |

*Test AUROC per class, selected model vs `base`:*

| <small>Model</small> | <small>NORM</small> | <small>MI</small> | <small>STTC</small> | <small>CD</small> | <small>HYP</small> |
|:------|-----:|---:|-----:|---:|----:|
| <small>`base`</small> | <small>$\color{#C0392B}{0.9489}$</small> | <small>$\color{#C0392B}{0.9379}$</small> | <small>$\color{#C0392B}{0.9350}$</small> | <small>$\color{#1E8449}{0.9274}$</small> | <small>$\color{#C0392B}{0.9031}$</small> |
| <small>`lww256d2dp`</small> | <small>$\color{#1E8449}{0.9520}$</small> | <small>$\color{#1E8449}{0.9405}$</small> | <small>$\color{#1E8449}{0.9362}$</small> | <small>$\color{#C0392B}{0.9270}$</small> | <small>$\color{#1E8449}{0.9082}$</small> |

- **The best single ResNet reaches 0.9328.** The selected ResNet reaches 0.9328 ± 0.0006 on the test set (0.9356 on validation). Across all 30 variants, test macro AUROC ranges from 0.928 to 0.934. For comparison, the best single model of the PTB-XL benchmark reaches 0.930 and its ensemble 0.934 [[4]](#references). Trained with the standard protocol, the selected model matches `resnet1d_wang` trained with the benchmark procedure (0.9324, best-validation checkpoint, B0).

- **A lead-wise stem is the only change that helps consistently.** Here the first layer processes each lead separately: a grouped convolution with 8 filters per lead, then a 1×1 convolution that combines the leads. This improves validation AUROC on all three seeds (+0.0007, +0.0017, +0.0006 over `base`). All five lead-wise variants are also at or above `base` on the test set (0.9308–0.9337). A plausible explanation is that the filters first learn per-lead waveform features. A standard stem, by contrast, mixes all 12 leads in its first convolution.

- **The selected model improves on `base` by 0.0023 on the test set,** and is higher on every seed (0.9321, 0.9334, 0.9328 compared with 0.9315, 0.9317, 0.9283). It also has a smaller spread over seeds. The gain is largest for HYP (+0.005), followed by NORM and MI (+0.003 each); CD is unchanged.

- **Most standard ResNet modifications do not help under this protocol.** Squeeze-excitation, pre-activation, ResNet-D, a deep stem, anti-aliasing, SiLU, other head dropout rates and bottleneck blocks remain within about 0.001 of `base` on validation, or fall below it. This includes a 19M-parameter 1D ResNet-50, 25 times the size of `base`. A larger receptive field (kernel 9, dilation) is consistently worse. More capacity alone (width 256 or more blocks) does not help; it helps slightly only in combination with stochastic depth (`w256d2dp`, +0.0006 on validation).

- **The validation ranking carried over to the test set, with the usual shrinkage.** Test values are 0.0015–0.0047 below validation for every variant. The variant with the highest test value (`lw16`, 0.9337) was not selected, because selection used validation only. Its difference from the selected model (0.0009) is within the variation between seeds.

- **The architecture search has saturated.** With the training procedure, the evaluation and the single-model constraint fixed, the ResNet levels off at about 0.933. Further gains, if any, must come from the factors studied in the other experiments: the input (leads, E4; sampling rate, E5), the augmentation (E3) or the operation type (E2).

<p align="center">
  <img src="runs/s2_resnet_search/figures/curves.png" alt="S2 learning curves" width="100%">
</p>

</details>

---

## References

1. Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W., and Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*, 7, 154. <https://doi.org/10.1038/s41597-020-0495-6>
2. Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., and Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.1). *PhysioNet*. <https://physionet.org/content/ptb-xl/1.0.1/>
3. Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., and Heldt, T. (2026). PhysioNet as a global platform for biomedical research. *Nature Health*. <https://doi.org/10.1038/s44360-026-00096-z>
4. Strodthoff, N., Wagner, P., Schaeffter, T., and Samek, W. (2021). Deep learning for ECG analysis: Benchmarks and insights from PTB-XL. *IEEE Journal of Biomedical and Health Informatics*, 25(5), 1519–1528. <https://doi.org/10.1109/JBHI.2020.3022989>
5. Loshchilov, I., and Hutter, F. (2019). Decoupled weight decay regularization. *International Conference on Learning Representations (ICLR)*. <https://arxiv.org/abs/1711.05101>
6. Smith, L. N., and Topin, N. (2019). Super-convergence: Very fast training of neural networks using large learning rates. *Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications*, Proc. SPIE 11006. <https://arxiv.org/abs/1708.07120>
7. Efron, B., and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.
8. He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep residual learning for image recognition. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778. <https://doi.org/10.1109/CVPR.2016.90>
9. Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., and He, K. (2017). Accurate, large minibatch SGD: Training ImageNet in 1 hour. <https://arxiv.org/abs/1706.02677>
10. Wang, Z., Yan, W., and Oates, T. (2017). Time series classification from scratch with deep neural networks: A strong baseline. *International Joint Conference on Neural Networks (IJCNN)*, 1578–1585. <https://arxiv.org/abs/1611.06455>
11. He, K., Zhang, X., Ren, S., and Sun, J. (2016). Identity mappings in deep residual networks. *European Conference on Computer Vision (ECCV)*, 630–645. <https://arxiv.org/abs/1603.05027>
12. Hu, J., Shen, L., and Sun, G. (2018). Squeeze-and-excitation networks. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 7132–7141. <https://arxiv.org/abs/1709.01507>
13. Huang, G., Sun, Y., Liu, Z., Sedra, D., and Weinberger, K. Q. (2016). Deep networks with stochastic depth. *European Conference on Computer Vision (ECCV)*, 646–661. <https://arxiv.org/abs/1603.09382>
14. He, T., Zhang, Z., Zhang, H., Zhang, Z., Xie, J., and Li, M. (2019). Bag of tricks for image classification with convolutional neural networks. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 558–567. <https://arxiv.org/abs/1812.01187>
15. Zhang, R. (2019). Making convolutional networks shift-invariant again. *International Conference on Machine Learning (ICML)*, 7324–7334. <https://arxiv.org/abs/1904.11486>
16. Yu, F., Koltun, V., and Funkhouser, T. (2017). Dilated residual networks. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 472–480. <https://arxiv.org/abs/1705.09914>
17. Hochreiter, S., and Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780. <https://doi.org/10.1162/neco.1997.9.8.1735>
18. Schuster, M., and Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, 45(11), 2673–2681. <https://doi.org/10.1109/78.650093>
19. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*, 5998–6008. <https://arxiv.org/abs/1706.03762>
20. Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., and Liu, T.-Y. (2020). On layer normalization in the Transformer architecture. *International Conference on Machine Learning (ICML)*, 10524–10533. <https://arxiv.org/abs/2002.04745>
21. Cortes, C., and Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297. <https://doi.org/10.1007/BF00994018>
