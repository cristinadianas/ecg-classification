"""E1 - Does depth help? ResNet1D-18 vs ResNet1D-34.

Both models are trained with the standard protocol (100 Hz, 12 leads, no augmentation,
50 epochs, official split) on the same seeds; only the number of residual blocks per
stage differs ([2,2,2,2] vs [3,4,6,3]). Finished runs are skipped, so the script can
be interrupted and restarted.

    python -m src.experiments.e1_depth                      # 2 models x 3 seeds
    python -m src.experiments.e1_depth --seeds 0            # quicker, one seed
    python -m src.experiments.e1_depth --epochs 2 --exp e1_smoke   # pipeline check

Any extra flag is passed to src.training.train. Results: runs/e1_depth/
(summary.md, runs.csv, figures/).
"""

import argparse

from src.config import STANDARD
from src.evaluation.summarize import write_summary
from src.training import train
from src.visualize.plot_runs import plot_experiment

EXP = "e1_depth"
MODELS = ["resnet1d18", "resnet1d34"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default=EXP)
    p.add_argument("--seeds", type=int, nargs="+", default=list(STANDARD["seeds"]))
    args, train_args = p.parse_known_args()

    # Seed-major order: after the first pass both models have one finished run.
    for seed in args.seeds:
        for model in MODELS:
            train.main(["--exp", args.exp, "--model", model, "--seeds", str(seed), *train_args])

    print()
    write_summary(args.exp)
    plot_experiment(args.exp)


if __name__ == "__main__":
    main()
