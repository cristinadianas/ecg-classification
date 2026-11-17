"""E2 - Which operation type fits ECG? Convolution vs BiLSTM vs CNN + Transformer.

Three model families that differ only in the operation that processes the sequence.
Everything else is shared: standard protocol + random crop (as since E3), the same
evaluation (maximum over sliding windows), the same MLP classifier on pooled features
(src/models/heads.py) and three seeds.

    convolution        ResNet (resnet1d_plus): the best ResNet of S1 and a wider, deeper one
    BiLSTM             bidirectional LSTM on the raw leads, as RNN1d of the benchmark (bilstm)
    CNN + Transformer  ResNet stem + 2 strided residual blocks (63 tokens per 2.5 s), then a
                       Transformer encoder (cnn_transformer)

Every family is trained in two sizes (about 0.7M and 2-5M parameters), so that a badly
chosen size does not decide the comparison. Within each family the size with the higher
mean validation macro AUROC is selected (validation only); the three selected models are
then compared on the test set (--show-test).

    python -m src.experiments.e2_operation                           # 6 models x 3 seeds
    python -m src.experiments.e2_operation --variants lstm_small_fp16 lstm_large_fp16  # float16-LSTM check
    python -m src.experiments.e2_operation --report-only --show-test # final report
    python -m src.experiments.e2_operation --epochs 2 --exp e2_smoke

Finished runs are skipped. Any extra flag is passed to src.training.train.
Results: runs/e2_operation/.
"""

import argparse
import json

import pandas as pd

from src.config import RUNS_DIR, STANDARD
from src.evaluation.summarize import collect_runs, write_summary
from src.training import train
from src.visualize.plot_runs import plot_experiment

EXP = "e2_operation"
AUGMENT = "crop"

# key -> (family, model, model kwargs, size tag, legend label); also the plot order
VARIANTS = {
    "conv_small": ("convolution", "resnet1d_plus", {}, "small", "ResNet, small (S1 best)"),
    "conv_large": ("convolution", "resnet1d_plus",
                   {"widths": [256] * 4, "layers": [2, 2, 2, 2], "drop_path": 0.2}, "large",
                   "ResNet, large (width 256, 2 blocks, stoch. depth)"),
    "lstm_small": ("BiLSTM", "bilstm", {"hidden": 128, "layers": 2}, "small", "BiLSTM, small (hidden 128)"),
    "lstm_large": ("BiLSTM", "bilstm", {"hidden": 256, "layers": 2}, "large", "BiLSTM, large (hidden 256)"),
    "trf_small": ("CNN + Transformer", "cnn_transformer", {"d_model": 128, "layers": 2, "heads": 4}, "small",
                  "CNN + Transformer, small (d 128, 2 layers)"),
    "trf_large": ("CNN + Transformer", "cnn_transformer", {"d_model": 256, "layers": 4, "heads": 8}, "large",
                  "CNN + Transformer, large (d 256, 4 layers)"),
}
# Diagnostic runs, not candidates: the BiLSTM with its recurrence in float16, as autocast
# runs cuDNN LSTMs by default. Only trained when named with --variants; never selected.
DIAGNOSTICS = {
    "lstm_small_fp16": ("BiLSTM", "bilstm", {"hidden": 128, "layers": 2, "fp32_lstm": False}, "small-fp16lstm",
                        "BiLSTM, small, float16 LSTM (diagnostic)"),
    "lstm_large_fp16": ("BiLSTM", "bilstm", {"hidden": 256, "layers": 2, "fp32_lstm": False}, "large-fp16lstm",
                        "BiLSTM, large, float16 LSTM (diagnostic)"),
}
ALL = {**VARIANTS, **DIAGNOSTICS}


def run(exp: str, keys: list[str], seeds: list[int], train_args: list[str]):
    # Seed-major order: after the first pass every variant has one finished run.
    for seed in seeds:
        for key in keys:
            _, model, kwargs, tag, _ = ALL[key]
            train.main(["--exp", exp, "--model", model, "--model-kwargs", json.dumps(kwargs),
                        "--augment", AUGMENT, "--tag", tag, "--seeds", str(seed), *train_args])


def report(exp: str, show_test: bool):
    runs = collect_runs(RUNS_DIR / exp)
    if runs.empty:
        return
    by_name = {(m, t): k for k, (_, m, _, t, _) in ALL.items()}
    runs["key"] = [by_name.get((m, v.rsplit("_", 1)[-1])) for m, v in zip(runs.model, runs.variant)]
    runs = runs.dropna(subset=["key"])
    g = runs.groupby("key")
    keys = [k for k in ALL if k in g.groups]
    table = pd.DataFrame({"family": [ALL[k][0] for k in keys], "label": [ALL[k][4] for k in keys],
                          "seeds": g.size()[keys].values, "params": g.n_params.first()[keys].values,
                          "train s": g.train_time_s.mean()[keys].round().values,
                          "best ep": g.best_epoch.mean()[keys].round(1).values}, index=keys)
    for c in ["val_macro_auc"] + (["test_macro_auc"] if show_test else []):
        table[f"{c} mean"] = g[c].mean()[keys]
        table[f"{c} std"] = g[c].std()[keys]
    candidates = table[table.index.isin(list(VARIANTS))]
    selected = candidates.groupby("family")["val_macro_auc mean"].idxmax()
    table["selected"] = table.index.isin(selected.values)
    print(f"\n{exp}: selection per family = mean validation macro AUROC" + ("" if show_test else " (test hidden)"))
    print(table.to_string(float_format="%.4f"))

    shown = [k for k in keys if k in VARIANTS]  # figures: candidates only
    variants = runs.drop_duplicates("key").set_index("key").variant
    plot_experiment(exp, variant_order=[variants[k] for k in shown],
                    labels={variants[k]: VARIANTS[k][4] for k in shown})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default=EXP)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(ALL),
                   help="default: the six candidates; the float16-LSTM diagnostics only when named")
    p.add_argument("--seeds", type=int, nargs="+", default=list(STANDARD["seeds"]))
    p.add_argument("--report-only", action="store_true", help="skip training, only summarize")
    p.add_argument("--show-test", action="store_true", help="also print test AUROC (final report only)")
    args, train_args = p.parse_known_args()
    if not args.report_only:
        run(args.exp, args.variants, args.seeds, train_args)
    write_summary(args.exp, show=args.show_test)
    report(args.exp, args.show_test)


if __name__ == "__main__":
    main()
