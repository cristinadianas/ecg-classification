"""B0: FCN baseline with the standard protocol.

    python -m src.experiments.b0_baseline
    python -m src.experiments.b0_baseline --seeds 0
"""

import argparse

from src.config import STANDARD
from src.evaluation.summarize import write_summary
from src.training import train
from src.visualize.plot_runs import plot_experiment

EXP = "b0_baseline"
MODEL = "fcn1d"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default=EXP)
    p.add_argument("--seeds", type=int, nargs="+", default=list(STANDARD["seeds"]))
    args, train_args = p.parse_known_args()

    for seed in args.seeds:
        train.main(["--exp", args.exp, "--model", MODEL, "--seeds", str(seed), *train_args])

    print()
    write_summary(args.exp)
    plot_experiment(args.exp)


if __name__ == "__main__":
    main()
