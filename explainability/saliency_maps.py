"""
Saliency map implementations:
  - Vanilla Gradients
  - Integrated Gradients
  - Guided Backpropagation (feature activation maps)
"""
import logging, uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)
HEATMAPS_DIR = Path("frontend/static/heatmaps"); HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)


def _save_map(arr: np.ndarray, prefix: str) -> str:
    colored = cv2.applyColorMap(
        cv2.resize(np.uint8(arr * 255), (224, 224)), cv2.COLORMAP_HOT
    )
    fname = f"{prefix}_{uuid.uuid4().hex[:10]}.png"
    cv2.imwrite(str(HEATMAPS_DIR / fname), colored)
    return f"static/heatmaps/{fname}"


def vanilla_saliency(
    model: tf.keras.Model,
    image_tensor: np.ndarray,         # (1,28,28,1)
) -> Optional[str]:
    try:
        x = tf.Variable(image_tensor, dtype=tf.float32)
        with tf.GradientTape() as tape:
            preds = model(x)
            top_c = int(tf.argmax(preds[0]))
            score = preds[:, top_c]
        grads    = tape.gradient(score, x)[0, :, :, 0].numpy()
        saliency = np.abs(grads)
        saliency = (saliency - saliency.min()) / (saliency.max() + 1e-8)
        return _save_map(saliency, "saliency_vanilla")
    except Exception as exc:
        logger.error(f"Vanilla saliency failed: {exc}")
        return None


def integrated_gradients(
    model: tf.keras.Model,
    image_tensor: np.ndarray,         # (1,28,28,1)
    steps: int = 50,
) -> Optional[str]:
    try:
        baseline    = np.zeros_like(image_tensor)
        alphas      = np.linspace(0, 1, steps + 1)
        interpolated = [baseline + a * (image_tensor - baseline) for a in alphas]
        interpolated = np.concatenate(interpolated, axis=0)   # (steps+1, 28, 28, 1)

        x  = tf.Variable(interpolated, dtype=tf.float32)
        with tf.GradientTape() as tape:
            preds = model(x)
            top_c = int(tf.argmax(model.predict(image_tensor, verbose=0)[0]))
            score = preds[:, top_c]
        grads = tape.gradient(score, x).numpy()                # (steps+1, 28, 28, 1)
        avg_grads  = grads[:-1].mean(axis=0)
        int_grads  = (image_tensor[0] - baseline[0]) * avg_grads
        importance = np.abs(int_grads[:, :, 0])
        importance = (importance - importance.min()) / (importance.max() + 1e-8)
        return _save_map(importance, "saliency_intgrad")
    except Exception as exc:
        logger.error(f"Integrated gradients failed: {exc}")
        return None


def feature_activation_map(
    model: tf.keras.Model,
    image_tensor: np.ndarray,
    layer_index: int = -3,
) -> Optional[str]:
    """Visualise mean activation of an intermediate conv layer."""
    try:
        layer = model.layers[layer_index]
        extractor = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
        acts = extractor.predict(image_tensor, verbose=0)[0]   # (h, w, C)
        mean_act = acts.mean(axis=-1)
        mean_act = (mean_act - mean_act.min()) / (mean_act.max() + 1e-8)
        return _save_map(mean_act, "activation_map")
    except Exception as exc:
        logger.error(f"Feature activation map failed: {exc}")
        return None
