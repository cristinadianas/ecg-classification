"""E3 - Which augmentations help? none vs noise / baseline wander / scaling / time shift /
random crop / all combined.

Each model is trained with the standard protocol under seven training-data conditions:
no augmentation, each of the five augmentations alone, and all five combined
(definitions and strengths in src/data/augment.py). Noise, wander, scaling and shift
are applied to a record with probability 0.5; random crop is applied to every record,
and models trained with it are evaluated on sliding windows (per-class maximum).
Validation and test data are never augmented. Finished runs are skipped, so the
script can be interrupted and restarted, and runs for further models can be added to
the same experiment folder later.

    python -m src.experiments.e3_augment                           # ResNet1D-10, 7 x 3 runs
    python -m src.experiments.e3_augment --models resnet1d10 bilstm  # once more models exist
    python -m src.experiments.e3_augment --epochs 2 --exp e3_smoke   # pipeline check

Any extra flag is passed to src.training.train. Results: runs/e3_augment/
(summary.md, runs.csv, figures/).
"""

import argparse

from src.config import RUNS_DIR, STANDARD
from src.evaluation.summarize import collect_runs, write_summary
from src.models import MODELS
from src.training import train
from src.visualize.plot_runs import plot_experiment

EXP = "e3_augment"
ALL = "noise,wander,scale,shift,crop"
CONDITIONS = ["none", "noise", "wander", "scale", "shift", "crop", ALL]
LABELS = {"none": "none", "noise": "noise", "wander": "baseline wander", "scale": "scaling",
          "shift": "time shift", "crop": "random crop", ALL: "all combined"}


def plot(exp: str, model_order: list[str]):
    """Plot with readable legend labels, in the order of CONDITIONS (and models)."""
    runs = collect_runs(RUNS_DIR / exp).drop_duplicates("variant")
    if runs.empty:
        return
    models = model_order + sorted(set(runs.model) - set(model_order))
    rank = lambda r: (models.index(r.model),  # noqa: E731
                      CONDITIONS.index(r.augment) if r.augment in CONDITIONS else len(CONDITIONS))
    runs = runs.assign(_rank=runs.apply(rank, axis=1)).sort_values("_rank")
    several = runs.model.nunique() > 1
    labels = {r.variant: (f"{r.model} · " if several else "") + LABELS.get(r.augment, r.augment)
              for r in runs.itertuples()}
    plot_experiment(exp, variant_order=list(runs.variant), labels=labels)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default=EXP)
    p.add_argument("--models", nargs="+", default=["resnet1d10"], choices=sorted(MODELS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(STANDARD["seeds"]))
    args, train_args = p.parse_known_args()

    # Seed-major order: after the first pass every condition has one finished run.
    for seed in args.seeds:
        for model in args.models:
            for aug in CONDITIONS:
                train.main(["--exp", args.exp, "--model", model, "--augment", aug,
                            "--seeds", str(seed), *train_args])

    print()
    write_summary(args.exp)
    plot(args.exp, args.models)


if __name__ == "__main__":
    main()
