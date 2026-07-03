"""
LIME  – Local Interpretable Model-agnostic Explanations for digit images.
"""
import logging, uuid
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
import tensorflow as tf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

logger = logging.getLogger(__name__)
HEATMAPS_DIR = Path("frontend/static/heatmaps"); HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)


def _predict_fn_factory(model: tf.keras.Model) -> Callable:
    """Return a function that accepts (N, 28, 28, 3) RGB float64 and returns probabilities."""
    def predict_fn(images):
        grey = images.mean(axis=-1, keepdims=True).astype("float32")
        return model.predict(grey, verbose=0)
    return predict_fn


def explain_lime(
    model: tf.keras.Model,
    image_tensor: np.ndarray,          # (1, 28, 28, 1)
    num_samples: int = 500,
    num_features: int = 5,
) -> Optional[str]:
    if not LIME_AVAILABLE:
        logger.warning("LIME not installed – skipping.")
        return None
    try:
        # LIME expects (H, W, 3) uint8
        grey_hw  = np.uint8(image_tensor[0, :, :, 0] * 255)
        rgb_hw   = np.repeat(grey_hw[:, :, np.newaxis], 3, axis=2)

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            rgb_hw,
            _predict_fn_factory(model),
            top_labels=1,
            hide_color=0,
            num_samples=num_samples,
            segmentation_fn=SegmentationAlgorithm(
                "quickshift", kernel_size=1, max_dist=200, ratio=0.2
            ),
        )
        top_label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            top_label,
            positive_only=True,
            num_features=num_features,
            hide_rest=False,
        )
        # Overlay mask on original
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(cv2.resize(grey_hw, (112, 112)), cmap="gray")
        ax.imshow(cv2.resize(mask.astype(float), (112, 112)),
                  alpha=0.5, cmap="Reds")
        ax.axis("off")
        ax.set_title(f"LIME – digit {top_label}", fontsize=9)
        fname = f"lime_{uuid.uuid4().hex[:10]}.png"
        fig.savefig(str(HEATMAPS_DIR / fname), dpi=100, bbox_inches="tight")
        plt.close(fig)
        return f"static/heatmaps/{fname}"
    except Exception as exc:
        logger.error(f"LIME explanation failed: {exc}")
        return None
