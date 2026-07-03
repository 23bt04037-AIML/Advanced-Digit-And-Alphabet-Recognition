r"""
Train digit CNN models using MNIST + local EMNIST digits + local SVHN.

Project root expected layout:
    E:\Week 2
    ├── datasets
    │   ├── EMNIST\gzip\emnist-digits-*.ubyte
    │   ├── train_32x32.mat
    │   └── test_32x32.mat
    ├── models
    └── training
        ├── train.py
        └── train_mixed_digits.py

Single model example:
    python training/train_mixed_digits.py --model cnn_medium --optimizer nadam --epochs 20 --save-alias

Faster first test:
    python training/train_mixed_digits.py --model cnn_medium --optimizer nadam --epochs 5 --max-emnist-train 10000 --max-svhn-train 5000 --save-alias
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Make imports work when running: python training/train_mixed_digits.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR = ROOT / "frontend" / "static" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
AUTOTUNE = tf.data.AUTOTUNE

# ---------------------------------------------------------------------
# Architectures and optimizers
# ---------------------------------------------------------------------
def _fallback_cnn_medium(input_shape=(28, 28, 1), num_classes=10):
    return keras.Sequential([
        layers.Input(input_shape),
        layers.Conv2D(32, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"),
        layers.Conv2D(32, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"),
        layers.MaxPooling2D(2), layers.Dropout(0.25),
        layers.Conv2D(64, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"),
        layers.Conv2D(64, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"),
        layers.MaxPooling2D(2), layers.Dropout(0.25),
        layers.Conv2D(128, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"), layers.Dropout(0.5),
        layers.Dense(128, activation="relu"), layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ], name="cnn_medium")


def _fallback_cnn_deep(input_shape=(28, 28, 1), num_classes=10):
    inp = layers.Input(input_shape)
    def conv_bn_relu(x, filters):
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        return layers.Activation("relu")(x)
    x = conv_bn_relu(inp, 32); x = conv_bn_relu(x, 32)
    x = layers.MaxPooling2D(2)(x); x = layers.Dropout(0.2)(x)
    x = conv_bn_relu(x, 64); x = conv_bn_relu(x, 64)
    x = layers.MaxPooling2D(2)(x); x = layers.Dropout(0.2)(x)
    x = conv_bn_relu(x, 128); x = conv_bn_relu(x, 128)
    x = layers.Dropout(0.2)(x)
    x = conv_bn_relu(x, 256)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x); x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x); x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="cnn_deep")

try:
    from training.train import BUILDERS as PROJECT_BUILDERS, OPTIMIZER_MAP as PROJECT_OPTIMIZERS
    BUILDERS = PROJECT_BUILDERS
    OPTIMIZER_MAP = PROJECT_OPTIMIZERS
except Exception:
    BUILDERS = {
        "cnn_medium": _fallback_cnn_medium,
        "cnn_deep": _fallback_cnn_deep,
    }
    OPTIMIZER_MAP = {
        "adam": lambda: keras.optimizers.Adam(1e-3),
        "rmsprop": lambda: keras.optimizers.RMSprop(1e-3),
        "nadam": lambda: keras.optimizers.Nadam(1e-3),
    }

# ---------------------------------------------------------------------
# IDX / EMNIST helpers
# ---------------------------------------------------------------------
def _is_gzip_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def _candidate_idx_files(path: Path):
    """
    Return possible IDX files for EMNIST.

    This handles all common Windows extraction cases:
    1) correct file: emnist-digits-train-images-idx3-ubyte
    2) gzip file:    emnist-digits-train-images-idx3-ubyte.gz
    3) folder made by extraction tool with the real file inside it
    """
    candidates = []

    def add(p: Path):
        try:
            if p.exists() and p.is_file():
                candidates.append(p)
        except OSError:
            pass

    add(path)
    add(Path(str(path) + ".gz"))

    # Your error came because this path exists as a DIRECTORY, not a file.
    # So search inside that directory for the actual IDX file.
    try:
        if path.exists() and path.is_dir():
            preferred_names = {path.name, path.name + ".gz"}
            inside = [p for p in path.rglob("*") if p.is_file()]
            candidates.extend([p for p in inside if p.name in preferred_names])
            candidates.extend([p for p in inside if p.name not in preferred_names])
    except OSError:
        pass

    # Also search parent folder for same base name.
    try:
        if path.parent.exists():
            for p in path.parent.glob(path.name + "*"):
                add(p)
    except OSError:
        pass

    # Remove duplicates while preserving order.
    seen = set()
    unique = []
    for p in candidates:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _open_idx(path: Path):
    candidates = _candidate_idx_files(path)
    if not candidates:
        raise FileNotFoundError(
            f"EMNIST IDX file not found for: {path}\n"
            f"Expected either this file, this .gz file, or a folder containing the real IDX file."
        )

    last_error = None
    for candidate in candidates:
        try:
            if candidate.suffix.lower() == ".gz" or _is_gzip_file(candidate):
                return gzip.open(candidate, "rb")
            return open(candidate, "rb")
        except PermissionError as e:
            last_error = e
            continue
        except OSError as e:
            last_error = e
            continue

    raise PermissionError(
        f"Could not open EMNIST IDX file for {path}. Tried: {[str(c) for c in candidates]}. "
        f"Last error: {last_error}"
    )


def read_idx_images(path: Path) -> np.ndarray:
    with _open_idx(path) as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image IDX magic number {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with _open_idx(path) as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label IDX magic number {magic} in {path}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.reshape(n)


def fix_emnist_orientation(images: np.ndarray) -> np.ndarray:
    # Converts EMNIST orientation to normal MNIST orientation.
    images = np.transpose(images, (0, 2, 1))
    images = np.flip(images, axis=1)
    return images


def load_emnist_digits(emnist_dir: Path, max_train: Optional[int], max_test: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_images = read_idx_images(emnist_dir / "emnist-digits-train-images-idx3-ubyte")
    train_labels = read_idx_labels(emnist_dir / "emnist-digits-train-labels-idx1-ubyte")
    test_images = read_idx_images(emnist_dir / "emnist-digits-test-images-idx3-ubyte")
    test_labels = read_idx_labels(emnist_dir / "emnist-digits-test-labels-idx1-ubyte")

    train_images = fix_emnist_orientation(train_images)
    test_images = fix_emnist_orientation(test_images)

    if max_train and max_train > 0:
        train_images, train_labels = train_images[:max_train], train_labels[:max_train]
    if max_test and max_test > 0:
        test_images, test_labels = test_images[:max_test], test_labels[:max_test]

    x_train = train_images.astype("float32") / 255.0
    x_test = test_images.astype("float32") / 255.0
    return x_train[..., None], train_labels.astype("int64"), x_test[..., None], test_labels.astype("int64")

# ---------------------------------------------------------------------
# MNIST / SVHN helpers
# ---------------------------------------------------------------------
def load_mnist(max_train: Optional[int], max_test: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    if max_train and max_train > 0:
        x_train, y_train = x_train[:max_train], y_train[:max_train]
    if max_test and max_test > 0:
        x_test, y_test = x_test[:max_test], y_test[:max_test]
    return (
        x_train.astype("float32")[..., None] / 255.0,
        y_train.astype("int64"),
        x_test.astype("float32")[..., None] / 255.0,
        y_test.astype("int64"),
    )


def _svhn_to_gray28(x: np.ndarray) -> np.ndarray:
    # SVHN .mat shape is (32, 32, 3, N). Convert to (N, 28, 28, 1).
    x = np.transpose(x, (3, 0, 1, 2))
    out = np.empty((x.shape[0], 28, 28), dtype=np.float32)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    for i, img in enumerate(x):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        gray = clahe.apply(gray)
        out[i] = gray.astype("float32") / 255.0
    return out[..., None]


def load_svhn(datasets_dir: Path, max_train: Optional[int], max_test: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scipy.io import loadmat
    train_mat = datasets_dir / "train_32x32.mat"
    test_mat = datasets_dir / "test_32x32.mat"
    if not train_mat.exists() or not test_mat.exists():
        raise FileNotFoundError(f"SVHN files not found: {train_mat} and {test_mat}")

    tr = loadmat(str(train_mat))
    te = loadmat(str(test_mat))
    x_train = _svhn_to_gray28(tr["X"])
    y_train = tr["y"].reshape(-1).astype("int64")
    x_test = _svhn_to_gray28(te["X"])
    y_test = te["y"].reshape(-1).astype("int64")
    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0

    if max_train and max_train > 0:
        x_train, y_train = x_train[:max_train], y_train[:max_train]
    if max_test and max_test > 0:
        x_test, y_test = x_test[:max_test], y_test[:max_test]
    return x_train.astype("float32"), y_train, x_test.astype("float32"), y_test

# ---------------------------------------------------------------------
# Dataset merge / tf.data
# ---------------------------------------------------------------------
def _shuffle_pair(x: np.ndarray, y: np.ndarray, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    return x[idx], y[idx]


def load_all_datasets(args):
    x_train_parts, y_train_parts, x_test_parts, y_test_parts = [], [], [], []

    if args.use_mnist:
        print("\nLoading MNIST from keras.datasets.mnist ...")
        xtr, ytr, xte, yte = load_mnist(args.max_mnist_train, args.max_mnist_test)
        print(f"  MNIST train={xtr.shape}, test={xte.shape}")
        x_train_parts.append(xtr); y_train_parts.append(ytr)
        x_test_parts.append(xte); y_test_parts.append(yte)

    if args.use_emnist:
        print(f"\nLoading EMNIST digits from {args.emnist_dir} ...")
        xtr, ytr, xte, yte = load_emnist_digits(Path(args.emnist_dir), args.max_emnist_train, args.max_emnist_test)
        print(f"  EMNIST train={xtr.shape}, test={xte.shape}")
        x_train_parts.append(xtr); y_train_parts.append(ytr)
        x_test_parts.append(xte); y_test_parts.append(yte)

    if args.use_svhn:
        print(f"\nLoading SVHN from {args.datasets_dir} ...")
        xtr, ytr, xte, yte = load_svhn(Path(args.datasets_dir), args.max_svhn_train, args.max_svhn_test)
        print(f"  SVHN train={xtr.shape}, test={xte.shape}")
        x_train_parts.append(xtr); y_train_parts.append(ytr)
        x_test_parts.append(xte); y_test_parts.append(yte)

    if not x_train_parts:
        raise ValueError("No dataset selected. Enable at least one dataset.")

    x_train = np.concatenate(x_train_parts, axis=0).astype("float32")
    y_train = np.concatenate(y_train_parts, axis=0).astype("int64")
    x_test = np.concatenate(x_test_parts, axis=0).astype("float32")
    y_test = np.concatenate(y_test_parts, axis=0).astype("int64")

    x_train, y_train = _shuffle_pair(x_train, y_train, seed=args.seed)
    x_test, y_test = _shuffle_pair(x_test, y_test, seed=args.seed + 1)

    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    print("\nFinal merged dataset:")
    print(f"  Train: {x_train.shape}, labels: {y_train.shape}")
    print(f"  Test : {x_test.shape}, labels: {y_test.shape}")
    print("  Train distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
    return x_train, y_train, y_train_cat, x_test, y_test, y_test_cat


def make_train_ds(x, y_cat, batch_size: int, augment: bool):
    ds = tf.data.Dataset.from_tensor_slices((x, y_cat))
    ds = ds.shuffle(min(len(x), 30000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if augment:
        aug = keras.Sequential([
            layers.RandomRotation(0.08),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.18),
        ], name="digit_augmentation")
        ds = ds.map(lambda a, b: (aug(a, training=True), b), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

# ---------------------------------------------------------------------
# Train / evaluate one model
# ---------------------------------------------------------------------
def train_single(args) -> Dict[str, Any]:
    tf.keras.utils.set_random_seed(args.seed)
    x_train, y_train, y_train_cat, x_test, y_test, y_test_cat = load_all_datasets(args)
    return train_single_from_arrays(args, x_train, y_train, y_train_cat, x_test, y_test, y_test_cat)


def train_single_from_arrays(args, x_train, y_train, y_train_cat, x_test, y_test, y_test_cat) -> Dict[str, Any]:
    if args.model not in BUILDERS:
        raise ValueError(f"Unknown model {args.model}. Available: {list(BUILDERS.keys())}")
    if args.optimizer not in OPTIMIZER_MAP:
        raise ValueError(f"Unknown optimizer {args.optimizer}. Available: {list(OPTIMIZER_MAP.keys())}")

    model = BUILDERS[args.model](input_shape=(28, 28, 1), num_classes=10)
    model.compile(optimizer=OPTIMIZER_MAP[args.optimizer](), loss="categorical_crossentropy", metrics=["accuracy"])

    run_name = f"{args.model}_{args.optimizer}"
    out_model = MODELS_DIR / f"{run_name}.keras"
    best_model = MODELS_DIR / f"{run_name}_best.keras"
    print(f"\n{'=' * 70}\nTraining {run_name}\nSaving to {out_model}\n{'=' * 70}")
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(str(best_model), monitor="val_accuracy", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=args.patience, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]
    train_ds = make_train_ds(x_train, y_train_cat, args.batch_size, augment=not args.no_augment)

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=(x_test, y_test_cat),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0

    model.save(out_model)
    if args.save_alias:
        alias_model = MODELS_DIR / f"{args.model}.keras"
        shutil.copyfile(out_model, alias_model)
    else:
        alias_model = None

    proba = model.predict(x_test, batch_size=args.batch_size, verbose=0)
    y_pred = proba.argmax(axis=1)
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "model_file": str(out_model),
        "best_model_file": str(best_model),
        "alias_file": str(alias_model) if alias_model else None,
        "model": args.model,
        "optimizer": args.optimizer,
        "run_name": run_name,
        "accuracy": float(report["accuracy"]),
        "accuracy_percent": round(float(report["accuracy"]) * 100, 2),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "confusion_matrix": cm.tolist(),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "epochs_requested": int(args.epochs),
        "epochs_completed": len(history.history.get("loss", [])),
        "training_time_seconds": round(elapsed, 2),
        "datasets": {
            "mnist": bool(args.use_mnist),
            "emnist": bool(args.use_emnist),
            "svhn": bool(args.use_svhn),
        },
        "history": {k: [float(vv) for vv in v] for k, v in history.history.items()},
    }
    metrics_path = MODELS_DIR / f"{run_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("\nTraining completed.")
    print(f"Saved model  : {out_model}")
    print(f"Saved best   : {best_model}")
    if alias_model:
        print(f"Saved alias  : {alias_model}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Accuracy     : {metrics['accuracy_percent']:.2f}%")
    return metrics


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train one mixed digit model.")
    parser.add_argument("--model", default="cnn_medium", choices=list(BUILDERS.keys()))
    parser.add_argument("--optimizer", default="nadam", choices=list(OPTIMIZER_MAP.keys()))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--save-alias", action="store_true", help="Also save models/cnn_medium.keras or models/cnn_deep.keras")
    parser.add_argument("--datasets-dir", default=str(ROOT / "datasets"))
    parser.add_argument("--emnist-dir", default=str(ROOT / "datasets" / "EMNIST" / "gzip"))
    parser.add_argument("--use-mnist", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-emnist", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-svhn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-mnist-train", type=int, default=60000)
    parser.add_argument("--max-mnist-test", type=int, default=10000)
    parser.add_argument("--max-emnist-train", type=int, default=50000)
    parser.add_argument("--max-emnist-test", type=int, default=10000)
    parser.add_argument("--max-svhn-train", type=int, default=30000)
    parser.add_argument("--max-svhn-test", type=int, default=8000)
    return parser


if __name__ == "__main__":
    train_single(build_arg_parser().parse_args())
