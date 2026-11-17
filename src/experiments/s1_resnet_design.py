"""S1 - Which ResNet design choices help? Custom ResNet variants vs the classic ResNet1D-10.

All models are trained with the standard protocol and random crop (the best
augmentation of E3); only the architecture changes. Starting from the classic
ResNet1D-10, each custom variant changes one design choice (or all three):

    resnet1d10                    classic layout (baseline; same runs as E3 "random crop")
    resnet1d10_stem1              stride-1 stem without max-pool: 4x finer time axis
    resnet1d10_chead              head of the benchmark: max + avg pooling, BN, dropout
    resnet1d10_w128               constant width 128 instead of 64-128-256-512
    resnet1d10_stem1_chead_w128   all three
    resnet1d18, resnet1d34        classic, deeper: does E1 change once overfitting is removed?
    resnet1d_wang                 benchmark model under *our* protocol (fair comparison with B0)

Finished runs are skipped, so the script can be interrupted and restarted.

    python -m src.experiments.s1_resnet_design                 # 8 models x 3 seeds
    python -m src.experiments.s1_resnet_design --epochs 2 --exp s1_smoke

Any extra flag is passed to src.training.train. Results: runs/s1_resnet_design/.
"""

import argparse

from src.config import RUNS_DIR, STANDARD
from src.evaluation.summarize import collect_runs, write_summary
from src.training import train
from src.visualize.plot_runs import plot_experiment

EXP = "s1_resnet_design"
AUGMENT = "crop"
VARIANTS = {  # model -> legend label (also the plot order)
    "resnet1d10": "ResNet1D-10 (classic)",
    "resnet1d10_stem1": "+ stride-1 stem",
    "resnet1d10_chead": "+ concat-pool head",
    "resnet1d10_w128": "+ width 128",
    "resnet1d10_stem1_chead_w128": "+ all three",
    "resnet1d18": "ResNet1D-18 (classic)",
    "resnet1d34": "ResNet1D-34 (classic)",
    "resnet1d_wang": "resnet1d_wang",
}


def plot(exp: str):
    runs = collect_runs(RUNS_DIR / exp).drop_duplicates("variant")
    if runs.empty:
        return
    order = list(VARIANTS)
    runs = runs.assign(_rank=runs.model.map(lambda m: order.index(m) if m in order else len(order)))
    runs = runs.sort_values("_rank")
    labels = {r.variant: VARIANTS.get(r.model, r.model) for r in runs.itertuples()}
    plot_experiment(exp, variant_order=list(runs.variant), labels=labels)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default=EXP)
    p.add_argument("--models", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(STANDARD["seeds"]))
    args, train_args = p.parse_known_args()

    # Seed-major order: after the first pass every variant has one finished run.
    for seed in args.seeds:
        for model in args.models:
            train.main(["--exp", args.exp, "--model", model, "--augment", AUGMENT,
                        "--seeds", str(seed), *train_args])

    print()
    write_summary(args.exp)
    plot(args.exp)


if __name__ == "__main__":
    main()
