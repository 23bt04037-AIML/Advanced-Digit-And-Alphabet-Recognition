"""
Stratified K-Fold cross-validation for CNN models.
Usage: python -m evaluation.cross_validation --model cnn_small --folds 5
"""
import argparse, json, logging, time
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow import keras

from training.train import cnn_small, cnn_medium, cnn_deep, load_mnist

logger    = logging.getLogger(__name__)
MODELS_DIR= Path("models"); MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR = Path("frontend/static/plots"); PLOTS_DIR.mkdir(parents=True, exist_ok=True)

BUILDERS = {"cnn_small": cnn_small, "cnn_medium": cnn_medium, "cnn_deep": cnn_deep}


def run_cv(model_name: str, n_folds: int = 5, epochs: int = 10,
           subset: int = 10000) -> dict:
    """
    Run stratified K-fold CV on a subset of MNIST for speed.
    Returns dict with per-fold and aggregated metrics.
    """
    logging.basicConfig(level=logging.INFO)
    (x_tr, y_tr, _), _ = load_mnist()

    # Use a stratified subset
    from sklearn.utils import resample
    x_sub, y_sub = resample(x_tr, y_tr, n_samples=subset,
                             stratify=y_tr, random_state=42)

    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(x_sub, y_sub), 1):
        logger.info(f"Fold {fold_idx}/{n_folds} …")
        x_f_tr, x_f_val = x_sub[tr_idx], x_sub[val_idx]
        y_f_tr, y_f_val = y_sub[tr_idx], y_sub[val_idx]

        y_cat_tr  = keras.utils.to_categorical(y_f_tr, 10)
        y_cat_val = keras.utils.to_categorical(y_f_val, 10)

        model = BUILDERS[model_name]()
        model.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        t0 = time.time()
        model.fit(
            x_f_tr, y_cat_tr,
            validation_data=(x_f_val, y_cat_val),
            epochs=epochs, batch_size=128,
            callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
            verbose=0,
        )
        elapsed = time.time() - t0
        y_pred  = model.predict(x_f_val, verbose=0).argmax(1)
        acc = accuracy_score(y_f_val, y_pred)
        f1  = f1_score(y_f_val, y_pred, average="macro")
        fold_results.append({
            "fold": fold_idx, "accuracy": acc, "macro_f1": f1,
            "time_s": round(elapsed, 1)
        })
        logger.info(f"  acc={acc:.4f}  f1={f1:.4f}")
        tf.keras.backend.clear_session()

    df = pd.DataFrame(fold_results)
    aggregated = {
        "model":    model_name,
        "folds":    n_folds,
        "mean_accuracy": round(df["accuracy"].mean(), 4),
        "std_accuracy":  round(df["accuracy"].std(),  4),
        "mean_f1":       round(df["macro_f1"].mean(), 4),
        "std_f1":        round(df["macro_f1"].std(),  4),
        "per_fold":      fold_results,
    }

    # Save results
    out = MODELS_DIR / f"{model_name}_cv_{n_folds}fold.json"
    out.write_text(json.dumps(aggregated, indent=2))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["fold"], df["accuracy"], color="#4e79a7", label="Accuracy", alpha=0.8)
    ax.bar(df["fold"], df["macro_f1"],  color="#f28e2b", label="F1",       alpha=0.8,
           bottom=df["accuracy"] - df["accuracy"])  # side-by-side trick
    ax.axhline(aggregated["mean_accuracy"], color="#4e79a7", ls="--",
               label=f"Mean Acc={aggregated['mean_accuracy']:.3f}")
    ax.axhline(aggregated["mean_f1"],       color="#f28e2b", ls="--",
               label=f"Mean F1={aggregated['mean_f1']:.3f}")
    ax.set(title=f"{model_name} – {n_folds}-Fold Cross Validation",
           xlabel="Fold", ylabel="Score", ylim=(0.9, 1.01), xticks=df["fold"])
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{model_name}_cv.png", dpi=130)
    plt.close()

    logger.info(f"\nCV Summary for {model_name}:")
    logger.info(f"  Acc: {aggregated['mean_accuracy']} ± {aggregated['std_accuracy']}")
    logger.info(f"  F1:  {aggregated['mean_f1']}       ± {aggregated['std_f1']}")
    return aggregated


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default="cnn_small",
                    choices=list(BUILDERS.keys()))
    ap.add_argument("--folds",  type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    a = ap.parse_args()
    run_cv(a.model, a.folds, a.epochs)
