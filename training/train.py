"""
Training pipeline for CNN digit recognition.

Supports:
- MNIST directly from TensorFlow internet download
- EMNIST local IDX files
- SVHN local .mat files
- Custom image dataset
- Combined training: MNIST + EMNIST + SVHN

Example commands:

MNIST + EMNIST + SVHN:
python training/train.py --dataset all --emnist-path "datasets/EMNIST/gzip" --svhn-path "datasets" --model cnn_deep --optimizer adamw --epochs 30 --batch-size 128 --image-size 32 --rgb

Only MNIST:
python training/train.py --dataset mnist --model cnn_deep --optimizer adamw --epochs 30

Only EMNIST:
python training/train.py --dataset emnist --emnist-path "datasets/EMNIST/gzip" --model cnn_deep --optimizer adamw --epochs 30

Only SVHN:
python training/train.py --dataset svhn --svhn-path "datasets" --model cnn_deep --optimizer adamw --epochs 30
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from tensorflow import keras
from tensorflow.keras import layers

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from preprocessing.preprocess import (
        build_augmentation_layer,
        load_custom_dataset,
        load_mnist,
        load_emnist,
        load_svhn,
    )
except Exception:
    from preprocess import (
        build_augmentation_layer,
        load_custom_dataset,
        load_mnist,
        load_emnist,
        load_svhn,
    )


# ============================================================
# LOGGING AND PATHS
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

PLOTS_DIR = Path("frontend/static/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# DATA PIPELINE
# ============================================================

def make_augmentation_dataset(x, y_cat, batch_size=128, augment=True):
    ds = tf.data.Dataset.from_tensor_slices((x, y_cat))
    ds = ds.shuffle(min(len(x), 20000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)

    if augment:
        aug = build_augmentation_layer()
        ds = ds.map(
            lambda imgs, labels: (aug(imgs, training=True), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def merge_datasets(dataset_list):
    x_train = np.concatenate([d[0][0] for d in dataset_list], axis=0)
    y_train = np.concatenate([d[0][1] for d in dataset_list], axis=0)

    x_test = np.concatenate([d[1][0] for d in dataset_list], axis=0)
    y_test = np.concatenate([d[1][1] for d in dataset_list], axis=0)

    train_idx = np.random.permutation(len(x_train))
    test_idx = np.random.permutation(len(x_test))

    x_train = x_train[train_idx]
    y_train = y_train[train_idx]

    x_test = x_test[test_idx]
    y_test = y_test[test_idx]

    return (x_train, y_train), (x_test, y_test)


def print_class_distribution(y, name):
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    logger.info(f"{name} class distribution: {dist}")


def load_selected_datasets(args):
    datasets_to_merge = []

    if args.dataset in ["mnist", "mnist_emnist", "mnist_svhn", "all"]:
        logger.info("Loading MNIST dataset...")
        datasets_to_merge.append(
            load_mnist(
                image_size=args.image_size,
                rgb=args.rgb,
                limit=args.limit_per_dataset,
            )
        )

    if args.dataset in ["emnist", "mnist_emnist", "emnist_svhn", "all"]:
        logger.info("Loading EMNIST dataset...")
        datasets_to_merge.append(
            load_emnist(
                emnist_path=args.emnist_path,
                subset=args.emnist_subset,
                image_size=args.image_size,
                rgb=args.rgb,
                limit=args.limit_per_dataset,
            )
        )

    if args.dataset in ["svhn", "mnist_svhn", "emnist_svhn", "all"]:
        logger.info("Loading SVHN dataset...")
        datasets_to_merge.append(
            load_svhn(
                svhn_path=args.svhn_path,
                image_size=args.image_size,
                rgb=args.rgb,
                limit=args.limit_per_dataset,
            )
        )

    if args.dataset == "custom":
        logger.info("Loading custom dataset...")
        datasets_to_merge.append(
            load_custom_dataset(
                args.dataset_path,
                image_size=args.image_size,
                rgb=args.rgb,
                limit=args.limit_per_dataset,
            )
        )

    if args.dataset == "both":
        logger.info("Loading MNIST + custom dataset...")
        datasets_to_merge.append(
            load_mnist(
                image_size=args.image_size,
                rgb=args.rgb,
                limit=args.limit_per_dataset,
            )
        )
        datasets_to_merge.append(
            load_custom_dataset(
                args.dataset_path,
                image_size=args.image_size,
                rgb=args.rgb,
                limit=args.limit_per_dataset,
            )
        )

    if len(datasets_to_merge) == 0:
        raise ValueError("No dataset selected.")

    if len(datasets_to_merge) == 1:
        return datasets_to_merge[0]

    return merge_datasets(datasets_to_merge)


# ============================================================
# CNN ARCHITECTURES
# ============================================================

def conv_block(x, filters, dropout=0.25):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout)(x)
    return x


def cnn_small(input_shape=(32, 32, 3), num_classes=10):
    inp = layers.Input(shape=input_shape)

    x = conv_block(inp, 32, 0.20)
    x = conv_block(x, 64, 0.25)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.40)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inp, out, name="cnn_small")


def cnn_medium(input_shape=(32, 32, 3), num_classes=10):
    inp = layers.Input(shape=input_shape)

    x = conv_block(inp, 32, 0.20)
    x = conv_block(x, 64, 0.25)
    x = conv_block(x, 128, 0.30)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.45)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.30)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inp, out, name="cnn_medium")


def cnn_deep(input_shape=(32, 32, 3), num_classes=10):
    inp = layers.Input(shape=input_shape)

    x = conv_block(inp, 32, 0.20)
    x = conv_block(x, 64, 0.25)
    x = conv_block(x, 128, 0.30)

    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.50)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.30)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inp, out, name="cnn_deep")


BUILDERS = {
    "cnn_small": cnn_small,
    "cnn_medium": cnn_medium,
    "cnn_deep": cnn_deep,
}

OPTIMIZER_CHOICES = ["adam", "adamw", "rmsprop", "nadam"]

MODEL_DEFAULT_EPOCHS = {
    "cnn_small": 15,
    "cnn_medium": 20,
    "cnn_deep": 25,
}


# ============================================================
# OPTIMIZERS
# ============================================================

def create_optimizer(name, learning_rate=1e-3, weight_decay=1e-4):
    name = name.lower()

    if name == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)

    if name == "adamw":
        if hasattr(keras.optimizers, "AdamW"):
            return keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )

        if hasattr(keras.optimizers, "experimental") and hasattr(
            keras.optimizers.experimental,
            "AdamW",
        ):
            return keras.optimizers.experimental.AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )

        raise RuntimeError(
            "AdamW is not available in your TensorFlow version. "
            "Use --optimizer adam or update TensorFlow."
        )

    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=learning_rate)

    if name == "nadam":
        return keras.optimizers.Nadam(learning_rate=learning_rate)

    raise ValueError(f"Unknown optimizer: {name}")


# ============================================================
# TRAINING
# ============================================================

def get_callbacks(run_name):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / f"{run_name}_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


def train_one(
    run_name,
    model_name,
    optimizer_name,
    epochs,
    batch_size,
    x_tr,
    y_tr_cat,
    x_te,
    y_te_cat,
    input_shape,
    augment=True,
    learning_rate=1e-3,
    weight_decay=1e-4,
    label_smoothing=0.05,
):
    model = BUILDERS[model_name](input_shape=input_shape, num_classes=10)

    optimizer = create_optimizer(
        optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing,
        ),
        metrics=["accuracy"],
    )

    logger.info(f"Model: {model_name}")
    logger.info(f"Optimizer: {optimizer_name}")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Training images: {len(x_tr)}")
    logger.info(f"Testing images: {len(x_te)}")
    logger.info(f"Augmentation: {augment}")

    train_ds = make_augmentation_dataset(
        x_tr,
        y_tr_cat,
        batch_size=batch_size,
        augment=augment,
    )

    t0 = time.time()

    hist = model.fit(
        train_ds,
        validation_data=(x_te, y_te_cat),
        epochs=epochs,
        callbacks=get_callbacks(run_name),
        verbose=1,
    )

    elapsed = time.time() - t0

    final_model_path = MODELS_DIR / f"{run_name}.keras"
    model.save(str(final_model_path))

    logger.info(f"Saved final model: {final_model_path}")
    logger.info(f"Training time: {elapsed:.1f} seconds")

    return model, hist, elapsed


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, run_name, x_te, y_te, batch_size=128):
    proba = model.predict(x_te, batch_size=batch_size, verbose=0)
    y_pred = proba.argmax(axis=1)

    report = classification_report(
        y_te,
        y_pred,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_te, y_pred, labels=list(range(10)))

    try:
        y_bin = label_binarize(y_te, classes=list(range(10)))
        auc_roc = roc_auc_score(
            y_bin,
            proba,
            multi_class="ovr",
            average="macro",
        )
    except Exception:
        auc_roc = float("nan")

    metrics = {
        "run_name": run_name,
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "auc_roc": float(auc_roc),
        "parameters": int(model.count_params()),
        "confusion_matrix": cm.tolist(),
    }

    metrics_path = MODELS_DIR / f"{run_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return metrics


# ============================================================
# PLOTS
# ============================================================

def plot_history(hist, run_name):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(hist.history["accuracy"], label="Train", linewidth=2)
    axes[0].plot(hist.history["val_accuracy"], label="Validation", linewidth=2)
    axes[0].set_title(f"{run_name} Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(hist.history["loss"], label="Train", linewidth=2)
    axes[1].plot(hist.history["val_loss"], label="Validation", linewidth=2)
    axes[1].set_title(f"{run_name} Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    save_path = PLOTS_DIR / f"{run_name}_training_curves.png"
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved training curve: {save_path}")


def plot_cm(cm_list, run_name):
    cm = np.array(cm_list)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm)

    ax.set_title(f"{run_name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    save_path = PLOTS_DIR / f"{run_name}_confusion_matrix.png"
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved confusion matrix: {save_path}")


# ============================================================
# GRID TRAINING
# ============================================================

def build_grid(model_filter, optimizer_filter, epochs, batch_size):
    models = list(BUILDERS.keys()) if model_filter == "all" else [model_filter]
    optimizers = OPTIMIZER_CHOICES if optimizer_filter == "all" else [optimizer_filter]

    configs = []

    for model_name in models:
        selected_epochs = MODEL_DEFAULT_EPOCHS[model_name] if epochs is None else epochs

        for optimizer_name in optimizers:
            configs.append(
                {
                    "model_name": model_name,
                    "optimizer_name": optimizer_name,
                    "epochs": selected_epochs,
                    "batch_size": batch_size,
                }
            )

    return configs


# ============================================================
# MAIN
# ============================================================

def main(args):
    set_seed(args.seed)

    logger.info("=" * 80)
    logger.info("DIGIT RECOGNITION CNN TRAINING")
    logger.info("=" * 80)

    logger.info(f"Selected dataset option: {args.dataset}")

    (x_tr, y_tr), (x_te, y_te) = load_selected_datasets(args)

    y_tr = y_tr.astype("int64")
    y_te = y_te.astype("int64")

    y_tr_cat = keras.utils.to_categorical(y_tr, 10)
    y_te_cat = keras.utils.to_categorical(y_te, 10)

    input_shape = x_tr.shape[1:]

    logger.info(f"Train shape: {x_tr.shape}")
    logger.info(f"Test shape: {x_te.shape}")
    logger.info(f"Input shape: {input_shape}")

    print_class_distribution(y_tr, "Train")
    print_class_distribution(y_te, "Test")

    configs = build_grid(
        model_filter=args.model,
        optimizer_filter=args.optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    all_metrics = []

    for cfg in configs:
        model_name = cfg["model_name"]
        optimizer_name = cfg["optimizer_name"]
        epochs = cfg["epochs"]
        batch_size = cfg["batch_size"]

        run_name = f"{args.dataset}_{model_name}_{optimizer_name}"

        logger.info("\n" + "=" * 80)
        logger.info(f"Training run: {run_name}")
        logger.info(f"Epochs: {epochs}")
        logger.info("=" * 80)

        model, hist, elapsed = train_one(
            run_name=run_name,
            model_name=model_name,
            optimizer_name=optimizer_name,
            epochs=epochs,
            batch_size=batch_size,
            x_tr=x_tr,
            y_tr_cat=y_tr_cat,
            x_te=x_te,
            y_te_cat=y_te_cat,
            input_shape=input_shape,
            augment=not args.no_augment,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
        )

        metrics = evaluate(
            model=model,
            run_name=run_name,
            x_te=x_te,
            y_te=y_te,
            batch_size=batch_size,
        )

        metrics["dataset"] = args.dataset
        metrics["model"] = model_name
        metrics["optimizer"] = optimizer_name
        metrics["epochs"] = epochs
        metrics["batch_size"] = batch_size
        metrics["image_size"] = args.image_size
        metrics["rgb"] = args.rgb
        metrics["augmentation"] = not args.no_augment
        metrics["training_time_seconds"] = round(elapsed, 1)

        all_metrics.append(metrics)

        plot_history(hist, run_name)
        plot_cm(metrics["confusion_matrix"], run_name)

        logger.info(
            f"Finished {run_name} | "
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"F1={metrics['macro_f1']:.4f} | "
            f"AUC={metrics['auc_roc']:.4f}"
        )

    summary_rows = []

    for m in all_metrics:
        row = {k: v for k, v in m.items() if k != "confusion_matrix"}
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)

    summary_path = MODELS_DIR / "training_summary.csv"
    df.to_csv(summary_path, index=False)

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\n{df.to_string(index=False)}")
    logger.info(f"Saved summary: {summary_path}")


# ============================================================
# ARGUMENTS
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CNN models on MNIST, EMNIST, SVHN, or custom digit dataset"
    )

    parser.add_argument(
        "--model",
        default="all",
        choices=list(BUILDERS.keys()) + ["all"],
        help="CNN model to train",
    )

    parser.add_argument(
        "--optimizer",
        default="all",
        choices=OPTIMIZER_CHOICES + ["all"],
        help="Optimizer to use",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Default: cnn_small=15, cnn_medium=20, cnn_deep=25",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size",
    )

    parser.add_argument(
        "--dataset",
        default="all",
        choices=[
            "mnist",
            "emnist",
            "svhn",
            "mnist_emnist",
            "mnist_svhn",
            "emnist_svhn",
            "all",
            "custom",
            "both",
        ],
        help="Dataset selection",
    )

    parser.add_argument(
        "--dataset-path",
        default=r"D:\DESKTOP\dataset\flattened_digit_dataset",
        help="Custom dataset path",
    )

    parser.add_argument(
        "--emnist-path",
        default=r"datasets\EMNIST\gzip",
        help="EMNIST gzip/IDX folder path",
    )

    parser.add_argument(
        "--emnist-subset",
        default="digits",
        choices=["digits", "mnist"],
        help="EMNIST subset for digit classification",
    )

    parser.add_argument(
        "--svhn-path",
        default=r"datasets",
        help="SVHN folder path containing train_32x32.mat and test_32x32.mat",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="Image size",
    )

    parser.add_argument(
        "--rgb",
        action="store_true",
        default=True,
        help="Use RGB images",
    )

    parser.add_argument(
        "--grayscale",
        dest="rgb",
        action="store_false",
        help="Use grayscale images",
    )

    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        help="Limit images per dataset for low RAM testing",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW",
    )

    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Label smoothing value",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation",
    )

    args = parser.parse_args()
    main(args)
