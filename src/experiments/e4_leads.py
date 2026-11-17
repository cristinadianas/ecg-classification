"""E4 - How many leads are needed? 12 leads vs 6 limb leads vs lead II only.

The reference model ResNet1D-10 (E1, E3) is trained with the standard protocol and
random 2.5 s crops (the best augmentation of E3) on three lead sets; everything else is
unchanged. Lead subsets are selected when the data are loaded (src/data/dataset.py),
so no new preprocessing is needed.

    all12   I, II, III, aVR, aVL, aVF, V1-V6
    limb6   I, II, III, aVR, aVL, aVF
    lead2   II

    python -m src.experiments.e4_leads                 # 3 lead sets x 3 seeds
    python -m src.experiments.e4_leads --epochs 2 --exp e4_smoke

Finished runs are skipped. Any extra flag is passed to src.training.train.
Results: runs/e4_leads/.
"""

import argparse

from src.config import STANDARD
from src.evaluation.summarize import write_summary
from src.training import train
from src.visualize.plot_runs import plot_experiment

EXP = "e4_leads"
MODEL = "resnet1d10"
AUGMENT = "crop"
LEAD_SETS = {  # --leads preset -> legend label (also the plot order)
    "all12": "12 leads",
    "limb6": "6 limb leads",
    "lead2": "lead II",
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default=EXP)
    p.add_argument("--leads", nargs="+", default=list(LEAD_SETS), choices=list(LEAD_SETS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(STANDARD["seeds"]))
    args, train_args = p.parse_known_args()

    # Seed-major order: after the first pass every lead set has one finished run.
    for seed in args.seeds:
        for leads in args.leads:
            train.main(["--exp", args.exp, "--model", MODEL, "--augment", AUGMENT, "--leads", leads,
                        "--seeds", str(seed), *train_args])

    print()
    write_summary(args.exp)
    labels = {f"{MODEL}_100hz_{leads}_aug-{AUGMENT}": label for leads, label in LEAD_SETS.items()}
    plot_experiment(args.exp, variant_order=list(labels), labels=labels)


if __name__ == "__main__":
    main()
