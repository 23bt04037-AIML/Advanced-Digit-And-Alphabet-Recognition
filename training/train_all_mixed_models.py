"""
Train multiple digit CNN models for >=95% target accuracy.

Default runs:
    cnn_medium: adam, rmsprop, nadam
    cnn_deep  : adam, rmsprop, nadam

Run from project root:
    python training/train_all_mixed_models.py --target-accuracy 0.95

For faster first run:
    python training/train_all_mixed_models.py --fast

For highest handwriting accuracy, train MNIST + EMNIST first:
    python training/train_all_mixed_models.py --no-use-svhn --target-accuracy 0.95
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import tensorflow as tf

from training.train_mixed_digits import (
    BUILDERS,
    OPTIMIZER_MAP,
    MODELS_DIR,
    load_all_datasets,
    train_single_from_arrays,
)


def _split_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _validate_names(models: List[str], optimizers: List[str]) -> None:
    bad_models = [m for m in models if m not in BUILDERS]
    bad_opts = [o for o in optimizers if o not in OPTIMIZER_MAP]
    if bad_models:
        raise ValueError(f"Unknown model(s): {bad_models}. Available: {list(BUILDERS.keys())}")
    if bad_opts:
        raise ValueError(f"Unknown optimizer(s): {bad_opts}. Available: {list(OPTIMIZER_MAP.keys())}")


def _make_single_args(base, model: str, optimizer: str):
    d = vars(base).copy()
    d["model"] = model
    d["optimizer"] = optimizer
    # Alias is assigned after all runs based on best optimizer for each architecture.
    d["save_alias"] = False
    return SimpleNamespace(**d)


def main(args):
    tf.keras.utils.set_random_seed(args.seed)

    models = _split_csv(args.models)
    optimizers = _split_csv(args.optimizers)
    _validate_names(models, optimizers)

    if args.fast:
        # Small subset for checking code works. Do not use this for final accuracy.
        args.max_mnist_train = min(args.max_mnist_train, 20000)
        args.max_mnist_test = min(args.max_mnist_test, 5000)
        args.max_emnist_train = min(args.max_emnist_train, 10000)
        args.max_emnist_test = min(args.max_emnist_test, 3000)
        args.max_svhn_train = min(args.max_svhn_train, 5000)
        args.max_svhn_test = min(args.max_svhn_test, 2000)
        args.epochs = min(args.epochs, 5)
        print("FAST MODE ENABLED: this is only for testing, not final accuracy.")

    print("\nLoading dataset once. All models will use the same data split...")
    x_train, y_train, y_train_cat, x_test, y_test, y_test_cat = load_all_datasets(args)

    results = []
    for model_name in models:
        for optimizer_name in optimizers:
            single_args = _make_single_args(args, model_name, optimizer_name)
            metrics = train_single_from_arrays(single_args, x_train, y_train, y_train_cat, x_test, y_test, y_test_cat)
            metrics["target_accuracy"] = float(args.target_accuracy)
            metrics["passed_target"] = bool(metrics["accuracy"] >= args.target_accuracy)
            results.append(metrics)

            status = "PASSED" if metrics["passed_target"] else "BELOW TARGET"
            print(f"\n{model_name}_{optimizer_name}: {metrics['accuracy_percent']:.2f}% -> {status}")

    summary_rows = []
    for m in results:
        summary_rows.append({
            "model": m["model"],
            "optimizer": m["optimizer"],
            "accuracy_percent": m["accuracy_percent"],
            "macro_f1": round(m["macro_f1"], 4),
            "epochs_completed": m["epochs_completed"],
            "train_samples": m["train_samples"],
            "test_samples": m["test_samples"],
            "passed_95_target": m["passed_target"],
            "model_file": m["model_file"],
        })

    df = pd.DataFrame(summary_rows).sort_values("accuracy_percent", ascending=False)
    summary_csv = MODELS_DIR / "all_mixed_models_summary.csv"
    summary_json = MODELS_DIR / "all_mixed_models_summary.json"
    df.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary_rows, indent=2))

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print(f"\nSaved summary CSV : {summary_csv}")
    print(f"Saved summary JSON: {summary_json}")

    # Save best alias per model architecture: cnn_medium.keras, cnn_deep.keras
    print("\nSaving best alias for each model architecture...")
    for model_name in models:
        model_df = df[df["model"] == model_name]
        if model_df.empty:
            continue
        best_row = model_df.iloc[0]
        src = Path(best_row["model_file"])
        dst = MODELS_DIR / f"{model_name}.keras"
        if src.exists():
            shutil.copyfile(src, dst)
            print(f"  {model_name}.keras <- {src.name} ({best_row['accuracy_percent']:.2f}%)")
        else:
            print(f"  Could not create alias for {model_name}; missing {src}")

    best_overall = df.iloc[0]
    src = Path(best_overall["model_file"])
    dst = MODELS_DIR / "best_digit_model.keras"
    if src.exists():
        shutil.copyfile(src, dst)
        print(f"\nBest overall alias: best_digit_model.keras <- {src.name} ({best_overall['accuracy_percent']:.2f}%)")

    if not bool(df["passed_95_target"].any()):
        print("\nWARNING: No model reached target accuracy.")
        print("Try these commands:")
        print("  1) For handwriting accuracy: python training/train_all_mixed_models.py --no-use-svhn --epochs 25 --target-accuracy 0.95")
        print("  2) For full mixed data: increase epochs and SVHN samples: python training/train_all_mixed_models.py --epochs 30 --max-svhn-train 60000 --target-accuracy 0.95")
        print("  3) Use the best saved model anyway; multi-digit accuracy also depends on segmentation/cropping.")


def build_parser():
    p = argparse.ArgumentParser(description="Train cnn_medium and cnn_deep with all optimizers on mixed digit datasets.")
    p.add_argument("--models", default="cnn_medium,cnn_deep", help="Comma separated. Example: cnn_medium,cnn_deep")
    p.add_argument("--optimizers", default="adam,rmsprop,nadam", help="Comma separated. Example: adam,rmsprop,nadam")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--target-accuracy", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--fast", action="store_true", help="Small subset test run. Not for final accuracy.")

    p.add_argument("--datasets-dir", default=str(ROOT / "datasets"))
    p.add_argument("--emnist-dir", default=str(ROOT / "datasets" / "EMNIST" / "gzip"))
    p.add_argument("--use-mnist", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-emnist", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-svhn", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--max-mnist-train", type=int, default=60000)
    p.add_argument("--max-mnist-test", type=int, default=10000)
    p.add_argument("--max-emnist-train", type=int, default=50000)
    p.add_argument("--max-emnist-test", type=int, default=10000)
    p.add_argument("--max-svhn-train", type=int, default=30000)
    p.add_argument("--max-svhn-test", type=int, default=8000)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
