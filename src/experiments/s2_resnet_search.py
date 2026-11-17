"""S2 - How far can a ResNet go? Architecture search within the ResNet family.

Starts from the best custom ResNet of S1 (stride-1 stem, concat-pooling head, constant
width 128; `base`) and varies the architecture (src/models/resnet1d_plus.py). Everything
else is fixed: one ResNet, trained with the standard protocol + random crop (as in S1)
and evaluated exactly as in E3/S1 (maximum over the sliding windows).

Model selection uses the VALIDATION set only (fold 9): with many variants, choosing by
test AUROC would overestimate the final result. Every variant is trained on three seeds
and ranked by its mean validation macro AUROC; the test set is printed only for the
final report (--show-test).

    python -m src.experiments.s2_resnet_search                              # all variants, seeds 0-2
    python -m src.experiments.s2_resnet_search --variants base dil --seeds 0
    python -m src.experiments.s2_resnet_search --report-only --show-test     # final report

Any extra flag is passed to src.training.train. Results: runs/s2_resnet_search/.
"""

import argparse
import json

import pandas as pd

from src.config import RUNS_DIR
from src.evaluation.summarize import collect_runs, write_summary
from src.training import train
from src.visualize.plot_runs import plot_experiment

EXP = "s2_resnet_search"
AUGMENT = "crop"

# tag -> resnet1d_plus kwargs (relative to the S1 best model) and legend label
VARIANTS = {
    "base":     ({}, "base (S1 best)"),
    # ---- round 1: one change at a time
    "se":       ({"se": True}, "+ squeeze-excitation"),
    "k3":       ({"kernel_size": 3}, "kernel 3"),
    "k7":       ({"kernel_size": 7}, "kernel 7"),
    "k9":       ({"kernel_size": 9}, "kernel 9"),
    "d2":       ({"layers": [2, 2, 2, 2]}, "2 blocks per stage"),
    "d2dp":     ({"layers": [2, 2, 2, 2], "drop_path": 0.2}, "2 blocks + stochastic depth"),
    "s3":       ({"layers": [1, 1, 1]}, "3 stages"),
    "w256":     ({"widths": [256] * 4}, "width 256"),
    "resd":     ({"resnet_d": True}, "ResNet-D shortcut"),
    "deepstem": ({"stem": "deep"}, "deep stem"),
    "preact":   ({"block": "preact"}, "pre-activation"),
    # ---- round 1: combinations (capacity + regularization)
    "w256d2dp": ({"widths": [256] * 4, "layers": [2, 2, 2, 2], "drop_path": 0.2},
                 "width 256 + 2 blocks + stochastic depth"),
    "sed2dp":   ({"se": True, "layers": [2, 2, 2, 2], "drop_path": 0.2},
                 "SE + 2 blocks + stochastic depth"),
    "s3w256":   ({"layers": [1, 1, 1], "widths": [256] * 3}, "3 stages + width 256"),
    # ---- round 2: receptive field, downsampling, stem, block type, head
    "dil":      ({"strides": [1, 1, 1, 1], "dilations": [1, 2, 4, 8]}, "dilated, no downsampling"),
    "dilrf":    ({"dilations": [1, 2, 2, 2]}, "dilation 2 (receptive field x1.8)"),
    "blur":     ({"downsample": "blur"}, "anti-aliased downsampling"),
    "silu":     ({"act": "silu"}, "SiLU activation"),
    "stemk15":  ({"stem_kernel_size": 15}, "stem kernel 15"),
    "leadwise": ({"stem": "leadwise"}, "lead-wise stem"),
    "k53":      ({"kernel_size": [5, 3]}, "kernels 5 + 3"),
    "w64":      ({"widths": [64] * 4}, "width 64"),
    "bneck50":  ({"block": "bottleneck", "layers": [3, 4, 6, 3], "widths": [256, 512, 1024, 2048]},
                 "bottleneck ResNet-50"),
    "ps25":     ({"ps_head": 0.25}, "head dropout 0.25"),
    "ps75":     ({"ps_head": 0.75}, "head dropout 0.75"),
    # ---- round 3: built on the best round-2 change (lead-wise stem, better than base on all seeds)
    "lw16":     ({"stem": "leadwise", "stem_per_lead": 16}, "lead-wise stem, 16 filters per lead"),
    "lwse":     ({"stem": "leadwise", "se": True}, "lead-wise stem + SE"),
    "lwsed2dp": ({"stem": "leadwise", "se": True, "layers": [2, 2, 2, 2], "drop_path": 0.2},
                 "lead-wise stem + SE + 2 blocks + stochastic depth"),
    "lww256d2dp": ({"stem": "leadwise", "widths": [256] * 4, "layers": [2, 2, 2, 2], "drop_path": 0.2},
                   "lead-wise stem + width 256 + 2 blocks + stochastic depth"),
}


def run(exp: str, tags: list[str], seeds: list[int], train_args: list[str]):
    for seed in seeds:
        for tag in tags:
            kwargs, _ = VARIANTS[tag]
            train.main(["--exp", exp, "--model", "resnet1d_plus", "--model-kwargs", json.dumps(kwargs),
                        "--augment", AUGMENT, "--tag", tag, "--seeds", str(seed), *train_args])


def report(exp: str, show_test: bool, n_plot: int = 5):
    runs = collect_runs(RUNS_DIR / exp)
    if runs.empty:
        return
    runs["tag"] = runs.variant.str.rsplit("_", n=1).str[-1]
    cols = ["val_macro_auc"] + (["test_macro_auc"] if show_test else [])
    g = runs.groupby("tag")
    table = pd.DataFrame({"label": [VARIANTS.get(t, ({}, t))[1] for t in g.groups],
                          "seeds": g.size().values}, index=list(g.groups))
    for c in cols:
        table[f"{c} mean"] = g[c].mean()
        table[f"{c} std"] = g[c].std()
    table["params"] = g.n_params.first()
    table = table.sort_values("val_macro_auc mean", ascending=False)
    print(f"\n{exp}: selection metric = mean validation macro AUROC" + ("" if show_test else " (test hidden)"))
    print(table.to_string(float_format="%.4f"))

    # figures: base and the n_plot best variants by validation
    order = ["base"] + [t for t in table.index if t != "base"][:n_plot]
    order = [t for t in order if t in set(runs.tag)]
    variants = runs.drop_duplicates("tag").set_index("tag").variant
    plot_experiment(exp, variant_order=[variants[t] for t in order],
                    labels={variants[t]: VARIANTS.get(t, ({}, t))[1] for t in order})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default=EXP)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--report-only", action="store_true", help="skip training, only summarize")
    p.add_argument("--show-test", action="store_true", help="also print test AUROC (final report only)")
    args, train_args = p.parse_known_args()
    if not args.report_only:
        run(args.exp, args.variants, args.seeds, train_args)
    write_summary(args.exp, show=args.show_test)
    report(args.exp, args.show_test)


if __name__ == "__main__":
    main()
