"""Paths, label/lead definitions and the standard training protocol.

Every experiment (B0, E1-E5) trains with STANDARD unless the experiment is
explicitly about one of these settings. Change a value here only if the whole team
agrees, because it invalidates comparisons with runs that are already done.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "ptb-xl"
PROCESSED_DIR = ROOT / "data" / "processed"
RUNS_DIR = ROOT / "runs"

# ----------------------------------------------------------------------------- labels

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# ----------------------------------------------------------------------------- signals

SAMPLING_RATES = (100, 500)
DURATION_S = 10

# Order of the 12 leads in the WFDB files, and therefore in the processed cache.
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Named lead subsets. Anything else can be passed as a comma-separated list
# (e.g. "I,II,V2"), see src.data.dataset.resolve_leads.
LEAD_SETS = {
    "all12": LEADS,
    "limb6": ["I", "II", "III", "aVR", "aVL", "aVF"],
    "precordial6": ["V1", "V2", "V3", "V4", "V5", "V6"],
    "lead1": ["I"],
    "lead2": ["II"],
}

# ----------------------------------------------------------------------------- split

# Official PTB-XL split (column strat_fold): patient-disjoint, label-stratified.
# Same as the Strodthoff et al. (2021) benchmark: train 1-8, validation 9, test 10.
TRAIN_FOLDS = (1, 2, 3, 4, 5, 6, 7, 8)
VAL_FOLDS = (9,)
TEST_FOLDS = (10,)

# ----------------------------------------------------------------------------- protocol

STANDARD = {
    # input
    "fs": 100,                  # Hz; E5 compares against 500
    "leads": "all12",           # E4 compares against limb6 / lead2
    "augment": "none",          # E3 compares augmentations
    # optimisation
    "epochs": 50,               # fixed for every run; no early stopping
    "batch_size": 128,
    "optimizer": "adamw",
    "lr": 1e-3,                 # peak learning rate of the one-cycle schedule
    "weight_decay": 1e-2,
    "scheduler": "onecycle",    # linear warm-up for 30% of steps, then cosine decay
    "pct_start": 0.3,
    "loss": "bce",              # BCEWithLogits on the 5 sigmoid outputs (multi-label)
    "dropout": 0.0,
    # model selection and reporting
    "selection_metric": "val_macro_auc",  # checkpoint kept = epoch with best val macro AUROC
    "threshold": 0.5,                     # F1 is reported at 0.5 and at val-tuned thresholds
    "seeds": (0, 1, 2),                   # report mean +- std over these seeds
    "bootstrap": 1000,                    # resamples for the test-set 95% CIs
}
