"""
Transfer learning for custom RGB digit dataset.
Updated for higher accuracy.

Examples:
    python transfer_learning.py --dataset custom --dataset-path "D:/DESKTOP/dataset/flattened_digit_dataset" --model mobilenetv2 --image-size 96
    python transfer_learning.py --dataset custom --dataset-path "D:/DESKTOP/dataset/flattened_digit_dataset" --model efficientnetb0 --image-size 96
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from tensorflow import keras
from tensorflow.keras import layers

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from preprocessing.preprocess import load_custom_dataset, load_mnist
except Exception:
    from preprocess import load_custom_dataset, load_mnist

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def preprocess_for_backbone(x, backbone):
    x = x.astype("float32")
    if backbone == "mobilenetv2":
        return keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
    if backbone == "resnet50":
        return keras.applications.resnet50.preprocess_input(x * 255.0)
    if backbone == "efficientnetb0":
        # EfficientNet preprocessing is included in the model in newer TF versions.
        return x
    return x


def build_mobilenetv2(input_shape, num_classes=10):
    base = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
    base.trainable = False
    inp = layers.Input(input_shape)
    x = base(inp, training=False)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="mobilenetv2")


def build_resnet50(input_shape, num_classes=10):
    base = keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
    base.trainable = False
    inp = layers.Input(input_shape)
    x = base(inp, training=False)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="resnet50")


def build_efficientnetb0(input_shape, num_classes=10):
    base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
    base.trainable = False
    inp = layers.Input(input_shape)
    x = base(inp, training=False)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="efficientnetb0")


BUILDERS = {
    "mobilenetv2": build_mobilenetv2,
    "resnet50": build_resnet50,
    "efficientnetb0": build_efficientnetb0,
}


def train_transfer(model, name, x_tr, y_tr_cat, x_te, y_te_cat, epochs_head=15, epochs_finetune=15, batch_size=64):
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        keras.callbacks.ModelCheckpoint(str(MODELS_DIR / f"{name}_best.keras"), monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    t0 = time.time()

    # Phase 1: train classifier head
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    model.fit(x_tr, y_tr_cat, validation_data=(x_te, y_te_cat), epochs=epochs_head, batch_size=batch_size, callbacks=callbacks, verbose=1)

    # Phase 2: fine-tune last layers
    base = model.layers[1]
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-5),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    model.fit(x_tr, y_tr_cat, validation_data=(x_te, y_te_cat), epochs=epochs_finetune, batch_size=batch_size, callbacks=callbacks, verbose=1)

    elapsed = time.time() - t0
    model.save(str(MODELS_DIR / f"{name}.keras"))
    return model, elapsed


def run(args):
    if args.dataset == "custom":
        (x_tr, y_tr), (x_te, y_te) = load_custom_dataset(args.dataset_path, image_size=args.image_size, rgb=True)
    elif args.dataset == "mnist":
        (x_tr, y_tr), (x_te, y_te) = load_mnist(image_size=args.image_size, rgb=True)
    elif args.dataset == "both":
        (x_tr_m, y_tr_m), (x_te_m, y_te_m) = load_mnist(image_size=args.image_size, rgb=True)
        (x_tr_c, y_tr_c), (x_te_c, y_te_c) = load_custom_dataset(args.dataset_path, image_size=args.image_size, rgb=True)
        x_tr = np.concatenate([x_tr_m, x_tr_c], axis=0)
        y_tr = np.concatenate([y_tr_m, y_tr_c], axis=0)
        x_te = np.concatenate([x_te_m, x_te_c], axis=0)
        y_te = np.concatenate([y_te_m, y_te_c], axis=0)

    y_tr_cat = keras.utils.to_categorical(y_tr, 10)
    y_te_cat = keras.utils.to_categorical(y_te, 10)

    input_shape = x_tr.shape[1:]
    logger.info(f"Train shape: {x_tr.shape} | Test shape: {x_te.shape}")

    x_tr = preprocess_for_backbone(x_tr, args.model)
    x_te = preprocess_for_backbone(x_te, args.model)

    todo = list(BUILDERS.keys()) if args.model == "all" else [args.model]

    results = []
    for model_name in todo:
        logger.info(f"Training {model_name}")
        model = BUILDERS[model_name](input_shape)
        model, elapsed = train_transfer(
            model,
            f"{args.dataset}_{model_name}",
            x_tr,
            y_tr_cat,
            x_te,
            y_te_cat,
            epochs_head=args.epochs_head,
            epochs_finetune=args.epochs_finetune,
            batch_size=args.batch_size,
        )

        proba = model.predict(x_te, verbose=0)
        y_pred = proba.argmax(axis=1)
        y_bin = label_binarize(y_te, classes=list(range(10)))

        metrics = {
            "model": model_name,
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "macro_f1": float(f1_score(y_te, y_pred, average="macro")),
            "auc_roc": float(roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")),
            "training_time_seconds": round(elapsed, 1),
        }
        results.append(metrics)
        (MODELS_DIR / f"{args.dataset}_{model_name}_metrics.json").write_text(json.dumps(metrics, indent=2))
        logger.info(json.dumps(metrics, indent=2))

    Path(MODELS_DIR / "transfer_learning_summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobilenetv2", choices=list(BUILDERS.keys()) + ["all"])
    parser.add_argument("--dataset", default="custom", choices=["custom", "mnist", "both"])
    parser.add_argument("--dataset-path", default=r"D:\DESKTOP\dataset\flattened_digit_dataset")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--epochs-head", type=int, default=15)
    parser.add_argument("--epochs-finetune", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    run(parser.parse_args())
