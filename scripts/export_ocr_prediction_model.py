"""
export_ocr_prediction_model.py

Add this file at:
    scripts/export_ocr_prediction_model.py

Run from project root:
    python scripts/export_ocr_prediction_model.py

Purpose:
    Your Kaggle file best_ocr_training_model.keras is a CTC training model.
    It has inputs:
        image, label, input_length, label_length
    and output:
        ctc_loss

    For Streamlit prediction, you need only:
        image -> ctc_softmax

This script extracts and saves:
    models/ocr_prediction_model.keras
"""

from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

TRAINING_MODEL_PATH = MODELS_DIR / "best_ocr_training_model.keras"
PREDICTION_MODEL_PATH = MODELS_DIR / "ocr_prediction_model.keras"
CHARSET_PATH = MODELS_DIR / "charset.json"

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def ctc_loss_func(args):
    labels_true, pred, input_length, label_length = args
    return K.ctc_batch_cost(labels_true, pred, input_length, label_length)


def main():
    if not TRAINING_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Training model not found: {TRAINING_MODEL_PATH}\n"
            "Put best_ocr_training_model.keras inside models/ first."
        )

    print("Loading training model:")
    print(TRAINING_MODEL_PATH)

    training_model = tf.keras.models.load_model(
        TRAINING_MODEL_PATH,
        compile=False,
        safe_mode=False,
        custom_objects={"ctc_loss_func": ctc_loss_func}
    )

    print("Training model loaded:", training_model.name)
    print("Inputs:", [x.name for x in training_model.inputs])
    print("Outputs:", [x.name for x in training_model.outputs])

    image_input = next((inp for inp in training_model.inputs if 'image' in inp.name.lower()), training_model.inputs[0])
    ocr_output = training_model.get_layer("ctc_softmax").output

    prediction_model = models.Model(
        inputs=image_input,
        outputs=ocr_output,
        name="ocr_prediction_model"
    )
    
    # Compile with None to clear the compile_config which contains the training CTC loss
    prediction_model.compile(optimizer="adam", loss=None)

    prediction_model.save(PREDICTION_MODEL_PATH, include_optimizer=False)

    print("Saved prediction model:")
    print(PREDICTION_MODEL_PATH)

    if not CHARSET_PATH.exists():
        cfg = {
            "charset": CHARSET,
            "blank_index": len(CHARSET),
            "num_classes": len(CHARSET) + 1,
            "img_h": 32,
            "img_w": 384,
            "max_label_len": 48,
        }

        with open(CHARSET_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)

        print("Saved charset config:")
        print(CHARSET_PATH)

    print("Done.")


if __name__ == "__main__":
    main()
