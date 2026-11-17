"""E2 (simple) - Convolution vs BiLSTM vs CNN + Transformer, one model per family.

The same question as E2 in its simplest form: one model per family, standard protocol,
random crop of 2.5 s as the only augmentation (evaluation: maximum over sliding
windows), three seeds.

    ResNet1D-10        the classic ResNet1D-10 of E1/E3 (resnet1d10)
    BiLSTM             hidden 128, 2 layers, recurrence in float16 (as autocast runs it)
    CNN + Transformer  d_model 128, 2 encoder layers, 4 heads

No model is selected here, so the test set is reported directly.

    python -m src.experiments.e2_operation_simple
    python -m src.experiments.e2_operation_simple --epochs 2 --exp e2_simple_smoke

Finished runs are skipped. Any extra flag is passed to src.training.train.
Results: runs/e2_operation_simple/.
"""

import argparse
import json

from src.config import STANDARD
from src.evaluation.summarize import write_summary
from src.training import train
from src.training.train import default_config, make_run_name
from src.visualize.plot_runs import plot_experiment

EXP = "e2_operation_simple"
AUGMENT = "crop"
MODELS = {  # key -> (model, model kwargs, run tag, legend label); also the plot order
    "resnet": ("resnet1d10", {}, "", "ResNet1D-10"),
    "bilstm": ("bilstm", {"hidden": 128, "layers": 2, "fp32_lstm": False}, "fp16lstm", "BiLSTM"),
    "transformer": ("cnn_transformer", {"d_model": 128, "layers": 2, "heads": 4}, "", "CNN + Transformer"),
}


def plot(exp: str):
    labels = {}
    for model, kwargs, tag, label in MODELS.values():
        variant = make_run_name(default_config(model=model, model_kwargs=kwargs, augment=AUGMENT, tag=tag))[1]
        labels[variant] = label
    plot_experiment(exp, variant_order=list(labels), labels=labels)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default=EXP)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(STANDARD["seeds"]))
    args, train_args = p.parse_known_args()

    # Seed-major order: after the first pass every model has one finished run.
    for seed in args.seeds:
        for key in args.models:
            model, kwargs, tag, _ = MODELS[key]
            train.main(["--exp", args.exp, "--model", model, "--model-kwargs", json.dumps(kwargs),
                        "--augment", AUGMENT, "--tag", tag, "--seeds", str(seed), *train_args])

    print()
    write_summary(args.exp)
    plot(args.exp)


if __name__ == "__main__":
    main()
