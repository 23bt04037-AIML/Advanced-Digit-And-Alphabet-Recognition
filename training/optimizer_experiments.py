"""
Systematic optimizer comparison experiment.
Trains the same medium CNN with Adam / RMSProp / Nadam / SGD and
produces a side-by-side accuracy/loss plot + CSV summary.

Usage: python -m training.optimizer_experiments
"""
import json, logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from training.train import load_mnist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
PLOTS_DIR = Path("frontend/static/plots"); PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(exist_ok=True)


OPTIMIZERS = {
    "Adam-1e-3":    keras.optimizers.Adam(1e-3),
    "Adam-1e-4":    keras.optimizers.Adam(1e-4),
    "RMSProp-1e-3": keras.optimizers.RMSprop(1e-3),
    "Nadam-1e-3":   keras.optimizers.Nadam(1e-3),
    "SGD-mom":      keras.optimizers.SGD(0.01, momentum=0.9, nesterov=True),
}

EPOCHS      = 15
BATCH_SIZE  = 128


def build_model():
    return keras.Sequential([
        layers.Input((28, 28, 1)),
        layers.Conv2D(32, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"),
        layers.Conv2D(64, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"),
        layers.MaxPooling2D(2), layers.Dropout(0.25),
        layers.Conv2D(128, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"), layers.Dropout(0.4),
        layers.Dense(10, activation="softmax"),
    ])


def run():
    (x_tr, y_tr, y_tr_cat), (x_te, y_te, y_te_cat) = load_mnist()
    histories, results = {}, []

    # On CPU: use a stratified 15k subset (matches train.py CPU strategy).
    # On GPU: use the full 60k dataset.
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        from sklearn.utils import resample
        n = 15_000
        x_tr, y_tr, y_tr_cat = resample(
            x_tr, y_tr, y_tr_cat, n_samples=n, stratify=y_tr, random_state=42)
        logger.info(f"CPU mode: using {n} training samples (stratified subset)")

    for name, opt in OPTIMIZERS.items():
        logger.info(f"Training with {name} …")
        model = build_model()
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
        hist = model.fit(x_tr, y_tr_cat,
                         validation_data=(x_te, y_te_cat),
                         epochs=EPOCHS, batch_size=BATCH_SIZE,
                         callbacks=[keras.callbacks.EarlyStopping(patience=4,
                                    restore_best_weights=True)],
                         verbose=0)
        histories[name] = hist.history
        proba  = model.predict(x_te, verbose=0)
        y_pred = proba.argmax(axis=1)
        results.append({
            "optimizer":  name,
            "best_val_acc": max(hist.history["val_accuracy"]),
            "final_loss":   hist.history["val_loss"][-1],
            "test_accuracy": accuracy_score(y_te, y_pred),
            "macro_f1":      f1_score(y_te, y_pred, average="macro"),
        })

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, h in histories.items():
        axes[0].plot(h["val_accuracy"], label=name, lw=2)
        axes[1].plot(h["val_loss"],     label=name, lw=2)
    axes[0].set(title="Val Accuracy per Optimizer", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
    axes[1].set(title="Val Loss per Optimizer",     xlabel="Epoch", ylabel="Loss")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "optimizer_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()

    df = pd.DataFrame(results)
    df.to_csv(MODELS_DIR / "optimizer_experiments.csv", index=False)
    print("\n" + df.to_string(index=False))
    logger.info("Done – results saved to models/optimizer_experiments.csv")


if __name__ == "__main__":
    run()
