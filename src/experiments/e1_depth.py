"""E1 - Does depth help? ResNet1D-10 vs ResNet1D-18 vs ResNet1D-34.

All models are trained with the standard protocol (100 Hz, 12 leads, no augmentation,
50 epochs, official split) on the same seeds; only the number of residual blocks per
stage differs ([1,1,1,1], [2,2,2,2], [3,4,6,3]). Finished runs are skipped, so the
script can be interrupted and restarted (e.g. after adding a model to MODELS).

    python -m src.experiments.e1_depth                      # 3 models x 3 seeds
    python -m src.experiments.e1_depth --seeds 0            # quicker, one seed
    python -m src.experiments.e1_depth --epochs 2 --exp e1_smoke   # pipeline check

Any extra flag is passed to src.training.train. Results: runs/e1_depth/
(summary.md, runs.csv, figures/).
"""

import argparse

from src.config import RUNS_DIR, STANDARD
from src.evaluation.summarize import collect_runs, write_summary
from src.training import train
from src.visualize.plot_runs import plot_experiment

EXP = "e1_depth"
MODELS = ["resnet1d10", "resnet1d18", "resnet1d34"]  # shallow -> deep (also the plot order)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default=EXP)
    p.add_argument("--seeds", type=int, nargs="+", default=list(STANDARD["seeds"]))
    args, train_args = p.parse_known_args()

    # Seed-major order: after the first pass every model has one finished run.
    for seed in args.seeds:
        for model in MODELS:
            train.main(["--exp", args.exp, "--model", model, "--seeds", str(seed), *train_args])

    print()
    write_summary(args.exp)
    runs = collect_runs(RUNS_DIR / args.exp).drop_duplicates("variant")
    order = sorted(runs.variant, key=lambda v: next(
        (i for i, m in enumerate(MODELS) if v.startswith(m + "_")), len(MODELS)))
    plot_experiment(args.exp, variant_order=order)


if __name__ == "__main__":
    main()
