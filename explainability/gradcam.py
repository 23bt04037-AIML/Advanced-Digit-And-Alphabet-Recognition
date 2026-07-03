"""
Grad-CAM  – Gradient-weighted Class Activation Mapping.
Produces heatmaps showing *which pixels* drove the model's decision.
"""
import logging, uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)
HEATMAPS_DIR = Path("frontend/static/heatmaps"); HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)


def compute_gradcam(
    model: tf.keras.Model,
    image_tensor: np.ndarray,
    class_index: Optional[int] = None,
    layer_name: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Returns a normalised float32 heatmap (H×W) for the target class.
    image_tensor: (1, 28, 28, 1)  float32  [0,1]
    """
    if layer_name is None:
        layer_name = next(
            (l.name for l in reversed(model.layers) if isinstance(l, tf.keras.layers.Conv2D)),
            None,
        )
    if layer_name is None:
        logger.warning("No Conv2D layer found; cannot compute Grad-CAM.")
        return None

    grad_model = tf.keras.Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        inputs_tf         = tf.cast(image_tensor, tf.float32)
        conv_outputs, preds = grad_model(inputs_tf)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        loss = preds[:, class_index]

    grads   = tape.gradient(loss, conv_outputs)                   # (1, h, w, C)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))              # (C,)
    cam     = tf.nn.relu(conv_outputs[0] @ weights[..., tf.newaxis])  # (h, w, 1)
    cam     = tf.squeeze(cam).numpy()
    cam    /= (cam.max() + 1e-8)
    return cam.astype("float32")


def overlay_gradcam(
    original_grey: np.ndarray,
    heatmap: np.ndarray,
    output_size: int = 224,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    original_grey: (28, 28) uint8  0–255
    heatmap:       (h, w)   float32 0–1
    Returns:       (output_size, output_size, 3) BGR uint8
    """
    orig_rgb   = cv2.cvtColor(
        cv2.resize(original_grey, (output_size, output_size)), cv2.COLOR_GRAY2BGR
    )
    h_resized  = cv2.resize(heatmap, (output_size, output_size))
    h_colored  = cv2.applyColorMap(np.uint8(255 * h_resized), cv2.COLORMAP_JET)
    overlay    = cv2.addWeighted(orig_rgb, 1 - alpha, h_colored, alpha, 0)
    return overlay


def generate_and_save(
    model: tf.keras.Model,
    image_tensor: np.ndarray,
    class_index: Optional[int] = None,
    layer_name: Optional[str] = None,
) -> Optional[str]:
    """High-level helper: compute → overlay → save → return URL path."""
    cam = compute_gradcam(model, image_tensor, class_index, layer_name)
    if cam is None:
        return None
    orig_grey = np.uint8(image_tensor[0, :, :, 0] * 255)
    overlay   = overlay_gradcam(orig_grey, cam)
    fname     = f"gradcam_{uuid.uuid4().hex[:10]}.png"
    cv2.imwrite(str(HEATMAPS_DIR / fname), overlay)
    return f"static/heatmaps/{fname}"
