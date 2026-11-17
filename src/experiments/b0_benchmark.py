"""B0 - Can we reproduce a benchmark? resnet1d_wang from Strodthoff et al. (2021).

Trains resnet1d_wang (src/models/resnet1d_wang.py) with the benchmark's training
procedure instead of our standard protocol. Differences from STANDARD:

    random 2.5 s crops for training, 7 sliding windows + per-class max for evaluation
    peak learning rate 1e-2 (standard: 1e-3)
    Adam beta2 0.99 (fastai default; standard: 0.999)
    one global mean/std for standardization (standard: per lead)

Shared with STANDARD: 100 Hz, 12 leads, folds 1-8 / 9 / 10, records without a
superclass dropped, BCE, AdamW with weight decay 1e-2, one-cycle, 50 epochs, batch 128.

The benchmark uses the model after the last epoch (no early stopping), so both
checkpoints are evaluated: last.pt (benchmark procedure, metrics_last.json) and
best.pt (our procedure, best validation macro AUROC, metrics.json).

    python -m src.experiments.b0_benchmark                     # 3 seeds
    python -m src.experiments.b0_benchmark --epochs 2 --exp b0_smoke

Any extra flag is passed to src.training.train. Results: runs/b0_benchmark/.
"""

import argparse
import json

import pandas as pd

from src.config import RUNS_DIR, STANDARD
from src.evaluation.evaluate import evaluate_run
from src.evaluation.summarize import collect_runs, write_summary
from src.training import train
from src.visualize.plot_runs import plot_experiment

EXP = "b0_benchmark"
REFERENCE_MACRO_AUC = 0.930  # Strodthoff et al. (2021), superdiagnostic, resnet1d_wang
BENCHMARK_ARGS = ["--model", "resnet1d_wang", "--augment", "crop", "--lr", "1e-2",
                  "--adam-beta2", "0.99", "--standardize", "global", "--tag", "benchmark"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default=EXP)
    p.add_argument("--seeds", type=int, nargs="+", default=list(STANDARD["seeds"]))
    args, train_args = p.parse_known_args()

    run_dirs = []
    for seed in args.seeds:
        run_dirs += train.main(["--exp", args.exp, *BENCHMARK_ARGS, "--seeds", str(seed), *train_args])
    for run_dir in run_dirs:  # the benchmark's checkpoint: the model after the last epoch
        if not (run_dir / "metrics_last.json").exists():
            cfg = json.loads((run_dir / "config.json").read_text())
            evaluate_run(run_dir, "last", n_boot=cfg.get("bootstrap"))

    print()
    write_summary(args.exp)
    runs = collect_runs(RUNS_DIR / args.exp)
    plot_experiment(args.exp, labels={v: "resnet1d_wang, benchmark procedure" for v in runs.variant})

    table = pd.DataFrame({
        "seed": runs.seed,
        "last epoch (benchmark)": runs.last_test_macro_auc,
        "best val epoch (ours)": runs.test_macro_auc,
        "best epoch": runs.best_epoch,
    })
    print(f"\nTest macro AUROC, resnet1d_wang, reference {REFERENCE_MACRO_AUC:.3f}:")
    print(table.to_string(index=False, float_format="%.4f"))
    print(f"mean ± std  last {runs.last_test_macro_auc.mean():.4f} ± {runs.last_test_macro_auc.std():.4f}"
          f"   best {runs.test_macro_auc.mean():.4f} ± {runs.test_macro_auc.std():.4f}")


if __name__ == "__main__":
    main()
