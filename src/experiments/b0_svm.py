"""B0: classical baseline - RBF SVM on hand-crafted features and on the raw waveform.

The three input representations of the original SVM notebooks (src/models/*.ipynb),
trained and evaluated in the standard pipeline on all 12 leads (src/training/train_svm.py):

    statistical  39 statistical/spectral features per lead (468 inputs)
    wavelet      62 ECG + wavelet features per lead (744 inputs)
    raw          the raw 12 x 1000 samples (12,000 inputs)

The SVM is deterministic, so one run per input is reported, with 95% bootstrap CIs
on the test set; further seeds reproduce it exactly (--seeds 0 1 2).

    python -m src.experiments.b0_svm
    python -m src.experiments.b0_svm --features wavelet

Finished runs are skipped. Results: runs/b0_svm/.
"""

import argparse

from src.evaluation.summarize import write_summary
from src.training import train_svm
from src.visualize.plot_runs import plot_experiment

EXP = "b0_svm"
VARIANTS = {  # --features -> legend label (also the plot order)
    "statistical": "SVM, statistical features",
    "wavelet": "SVM, ECG + wavelet features",
    "raw": "SVM, raw waveform",
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default=EXP)
    p.add_argument("--features", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    args, extra = p.parse_known_args()

    for seed in args.seeds:
        for features in args.features:
            train_svm.main(["--exp", args.exp, "--features", features, "--seeds", str(seed), *extra])

    print()
    write_summary(args.exp)
    labels = {f"svm-{f}_100hz_all12": label for f, label in VARIANTS.items()}
    plot_experiment(args.exp, variant_order=list(labels), labels=labels)


if __name__ == "__main__":
    main()
