"""
Core AI inference service:
  - Lazy model loading registry
  - Image preprocessing
  - Prediction with confidence + top-3
  - Grad-CAM heatmap generation
"""
import os, time, uuid, logging
import numpy as np
from PIL import Image, ImageOps
import cv2
import tensorflow as tf
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

MODELS_DIR   = Path(os.getenv("MODELS_DIR", "models"))
HEATMAPS_DIR = Path("frontend/static/heatmaps")
UPLOADS_DIR  = Path("frontend/static/uploads")
for d in (HEATMAPS_DIR, UPLOADS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ── Model Registry (singleton, lazy-load) ────────────────────────────────────
class _ModelRegistry:
    _models: Dict[str, tf.keras.Model] = {}

    def get(self, name: str) -> Optional[tf.keras.Model]:
        if name in self._models:
            return self._models[name]
        # Build candidate list: exact name first, then common optimizer suffixes
        candidates = [name] + [f"{name}_{opt}" for opt in ("nadam", "rmsprop", "adam", "sgd")]
        for candidate in candidates:
            for ext in (".keras", ".h5"):
                p = MODELS_DIR / f"{candidate}{ext}"
                if p.exists():
                    logger.info(f"Loading model: {p}  (requested: {name})")
                    m = tf.keras.models.load_model(str(p))
                    self._models[name] = m  # cache under the requested name
                    return m
        logger.error(f"Model '{name}' not found in {MODELS_DIR}")
        return None

    def available(self) -> List[str]:
        names = set()
        for ext in (".keras", ".h5"):
            for p in MODELS_DIR.glob(f"*{ext}"):
                names.add(p.stem)
        return sorted(names)


registry = _ModelRegistry()


# ── Image pre-processing ─────────────────────────────────────────────────────
def preprocess_image(
    img: np.ndarray,
    target_h: int = 28,
    target_w: int = 28,
    target_c: int = 1,
) -> np.ndarray:
    """
    Resize and normalise `img` to exactly (1, target_h, target_w, target_c).

    - target_c == 1  → convert to grayscale, normalise [0,1], MNIST polarity
    - target_c == 3  → keep/convert to BGR,  normalise [-1,1] (MobileNet style)
    """
    if target_c == 1:
        # Grayscale path (CNN models)
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img.astype(np.float32), (target_w, target_h),
                         interpolation=cv2.INTER_AREA)
        if img.mean() > 127:          # white-background → invert
            img = 255.0 - img
        img = img / 255.0
        return img.reshape(1, target_h, target_w, 1).astype(np.float32)
    else:
        # RGB / 3-channel path (transfer-learning models)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img.astype(np.float32), (target_w, target_h),
                         interpolation=cv2.INTER_AREA)
        img = (img / 127.5) - 1.0    # [-1, 1]  (MobileNetV2 / ResNet50)
        return img.reshape(1, target_h, target_w, target_c).astype(np.float32)


def _model_input_shape(model: tf.keras.Model):
    """Return (H, W, C) from the model's first input layer."""
    shape = model.input_shape          # e.g. (None, 32, 32, 3)
    return int(shape[1]), int(shape[2]), int(shape[3])


def pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image to numpy BGR/grey array with phone EXIF orientation fixed."""
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    return np.array(pil_img)[:, :, ::-1]  # PIL RGB -> CV2 BGR


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(image_array: np.ndarray, model_name: str = "cnn_medium") -> Dict[str, Any]:
    """
    Run inference and return structured result dict.
    Raises ValueError if model not found.
    """
    t0 = time.time()
    model = registry.get(model_name)
    if model is None:
        raise ValueError(f"Model '{model_name}' not available. Train it first.")

    h, w, c = _model_input_shape(model)
    tensor = preprocess_image(image_array, h, w, c)
    logger.info(f"[predict] model={model_name!r}  model_expects=({h},{w},{c})  "
                f"input.shape={image_array.shape}  tensor.shape={tensor.shape}")
    proba  = model.predict(tensor, verbose=0)[0]            # shape (10,)

    pred_idx = int(np.argmax(proba))
    confidence = float(proba[pred_idx])

    top3_idx = np.argsort(proba)[-3:][::-1]
    top3 = [{"digit": int(i), "confidence": round(float(proba[i]) * 100, 2)}
            for i in top3_idx]

    return {
        "predicted_digit":  pred_idx,
        "confidence":        round(confidence * 100, 2),
        "top3_predictions":  top3,
        "all_probabilities": [round(float(p) * 100, 2) for p in proba],
        "processing_time_ms": round((time.time() - t0) * 1000, 2),
        "model_used":         model_name,
    }


def multi_digit_predict(
    image_array: np.ndarray,
    model_name: str = "cnn_deep",
    confidence_threshold: float = 50.0,
    max_digits: int = 50,
) -> Dict[str, Any]:
    """
    Detect and predict all digits in one image.

    This is the correct pipeline for multi-digit images:
        full image -> segmentation -> 28x28 digit crops -> CNN prediction per crop

    Returns a dictionary that Streamlit/API can display directly.
    """
    t0 = time.time()

    # Lazy import avoids circular imports and keeps single-digit prediction unchanged.
    from preprocessing.multidigit import (
        auto_orient_for_multidigit_line,
        detect_and_prepare_digits,
        draw_digit_boxes,
    )

    # Fix common phone-photo orientation issue before detection/annotation.
    image_array = auto_orient_for_multidigit_line(image_array)

    candidates, mask, enhanced_gray = detect_and_prepare_digits(image_array, max_digits=max_digits)
    if not candidates:
        return {
            "success": False,
            "prediction": "",
            "digits": [],
            "message": "No digits detected. Try a clearer image or crop the number area.",
            "processing_time_ms": round((time.time() - t0) * 1000, 2),
            "model_used": model_name,
            "mask_image": mask,
            "enhanced_gray": enhanced_gray,
            "annotated_image": image_array,
        }

    digit_results: List[Dict[str, Any]] = []
    final_digits: List[str] = []

    for pos, candidate in enumerate(candidates, start=1):
        # The existing predict() expects a normal image array. Send BGR 28x28 canvas.
        digit_bgr = cv2.cvtColor(candidate.canvas28, cv2.COLOR_GRAY2BGR)
        pred = predict(digit_bgr, model_name=model_name)

        digit = int(pred["predicted_digit"])
        confidence = float(pred["confidence"])
        status = "ok" if confidence >= confidence_threshold else "low_confidence"

        final_digits.append(str(digit) if status == "ok" else "?")
        digit_results.append({
            "position": pos,
            "digit": digit,
            "confidence": round(confidence, 2),
            "status": status,
            "box": tuple(int(v) for v in candidate.box),
            "top3_predictions": pred.get("top3_predictions", []),
            "canvas28": candidate.canvas28,
        })

    annotated = draw_digit_boxes(image_array, digit_results)
    uncertain_count = sum(1 for d in digit_results if d["status"] != "ok")

    return {
        "success": True,
        "prediction": "".join(final_digits),
        "raw_prediction": "".join(str(d["digit"]) for d in digit_results),
        "digits": digit_results,
        "digit_count": len(digit_results),
        "uncertain_count": uncertain_count,
        "message": "OK" if uncertain_count == 0 else f"{uncertain_count} digit(s) have low confidence.",
        "processing_time_ms": round((time.time() - t0) * 1000, 2),
        "model_used": model_name,
        "mask_image": mask,
        "enhanced_gray": enhanced_gray,
        "annotated_image": annotated,
    }


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def generate_gradcam(
    image_array: np.ndarray,
    model_name: str = "cnn_medium",
    pred_index: Optional[int] = None,
) -> Optional[str]:
    """
    Generate Grad-CAM overlay using fully eager execution (no sub-model build).
    Works with both Functional and Sequential Keras models.
    Returns None on any failure (non-fatal).
    """
    try:
        model = registry.get(model_name)
        if model is None:
            return None

        h, w, c = _model_input_shape(model)
        tensor = tf.cast(preprocess_image(image_array, h, w, c), tf.float32)

        # Find last Conv2D layer index
        conv_layer_idx = None
        conv_layer = None
        for i, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_layer_idx = i
                conv_layer = layer

        if conv_layer is None:
            logger.warning("No Conv2D layer found; skipping Grad-CAM.")
            return None

        # Run eagerly: step through layers manually, record conv output
        with tf.GradientTape() as tape:
            x = tensor
            conv_out = None
            for i, layer in enumerate(model.layers):
                x = layer(x, training=False)
                if i == conv_layer_idx:
                    conv_out = x
                    tape.watch(conv_out)
            preds = x  # final output after all layers

            if pred_index is None:
                pred_index = int(tf.argmax(preds[0]))
            class_score = preds[:, pred_index]

        grads   = tape.gradient(class_score, conv_out)
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.nn.relu(conv_out[0] @ pooled[..., tf.newaxis])
        heatmap = tf.squeeze(heatmap).numpy()
        if heatmap.ndim == 0:
            heatmap = np.array([[float(heatmap)]])
        heatmap /= (heatmap.max() + 1e-8)

        # Build overlay
        if c == 3:
            orig_rgb = np.uint8((tensor.numpy()[0] + 1.0) * 127.5)
        else:
            orig_grey = (tensor.numpy()[0, :, :, 0] * 255).astype(np.uint8)
            orig_rgb  = cv2.cvtColor(orig_grey, cv2.COLOR_GRAY2RGB)

        h_resized = cv2.resize(heatmap, (w, h))
        h_colored = cv2.applyColorMap(np.uint8(255 * h_resized), cv2.COLORMAP_JET)
        overlay   = cv2.addWeighted(orig_rgb, 0.55, h_colored, 0.45, 0)
        overlay   = cv2.resize(overlay, (224, 224), interpolation=cv2.INTER_NEAREST)

        fname    = f"gradcam_{uuid.uuid4().hex[:10]}.png"
        out_path = HEATMAPS_DIR / fname
        cv2.imwrite(str(out_path), overlay)
        return f"static/heatmaps/{fname}"

    except Exception as exc:
        logger.error(f"Grad-CAM failed: {exc}")
        return None


# ── Saliency Map (vanilla gradients) ─────────────────────────────────────────
def generate_saliency(
    image_array: np.ndarray,
    model_name: str = "cnn_medium",
) -> Optional[str]:
    """Vanilla gradient saliency map."""
    try:
        model = registry.get(model_name)
        if model is None:
            return None

        h, w, c = _model_input_shape(model)
        tensor = tf.Variable(preprocess_image(image_array, h, w, c), dtype=tf.float32)
        with tf.GradientTape() as tape:
            preds = model(tensor)
            top_class = tf.argmax(preds[0])
            class_score = preds[:, top_class]

        # Average over channel dim for multi-channel tensors
        g = tape.gradient(class_score, tensor)[0].numpy()  # (H, W, C) or (H, W)
        grads    = g.mean(axis=-1) if g.ndim == 3 else g
        saliency = np.abs(grads)
        saliency = (saliency - saliency.min()) / (saliency.max() + 1e-8)
        saliency_img = cv2.applyColorMap(
            np.uint8(255 * cv2.resize(saliency, (224, 224))), cv2.COLORMAP_HOT
        )
        fname    = f"saliency_{uuid.uuid4().hex[:10]}.png"
        out_path = HEATMAPS_DIR / fname
        cv2.imwrite(str(out_path), saliency_img)
        return f"static/heatmaps/{fname}"

    except Exception as exc:
        logger.error(f"Saliency map failed: {exc}")
        return None


# ── LIME (Local Interpretable Model-Agnostic Explanations) ─────────────────────
def generate_lime(
    image_array: np.ndarray,
    model_name: str = "cnn_medium",
) -> Optional[str]:
    """Generate LIME explanation overlay."""
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries

        model = registry.get(model_name)
        if model is None:
            return None

        h, w, c = _model_input_shape(model)
        tensor = preprocess_image(image_array, h, w, c)

        def predict_fn(images):
            # images shape: (batch_size, H, W, 3) 
            batch = []
            for img in images:
                if img.max() > 1.0:
                    img = img / 255.0
                if c == 1:
                    gray = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2GRAY)
                    batch.append(gray.reshape(h, w, 1))
                else:
                    batch.append((img * 2.0) - 1.0)
            batch = np.array(batch, dtype=np.float32)
            return model.predict(batch, verbose=0)

        explainer = lime_image.LimeImageExplainer()
        
        # Prepare input for explainer (H, W, 3) [0, 1]
        if c == 3:
            lime_input = (tensor[0] + 1.0) / 2.0
        else:
            lime_input = np.repeat(tensor[0], 3, axis=-1)
            
        explanation = explainer.explain_instance(
            lime_input.astype('double'), 
            predict_fn, 
            top_labels=1, 
            hide_color=0, 
            num_samples=300  # Kept small for speed
        )
        
        top_pred = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(top_pred, positive_only=False, num_features=5, hide_rest=False)
        
        # Draw boundaries
        img_boundry = mark_boundaries(temp, mask)
        img_boundry = (img_boundry * 255).astype(np.uint8)
        img_boundry = cv2.cvtColor(img_boundry, cv2.COLOR_RGB2BGR)
        
        overlay = cv2.resize(img_boundry, (224, 224), interpolation=cv2.INTER_NEAREST)

        fname    = f"lime_{uuid.uuid4().hex[:10]}.png"
        out_path = HEATMAPS_DIR / fname
        cv2.imwrite(str(out_path), overlay)
        return f"static/heatmaps/{fname}"
        
    except Exception as exc:
        logger.error(f"LIME failed: {exc}")
        return None


# ── FGSM Adversarial Attack ──────────────────────────────────────────────────
def generate_fgsm_attack(
    image_array: np.ndarray,
    epsilon: float,
    model_name: str = "cnn_medium",
) -> Optional[Dict[str, Any]]:
    """
    Apply FGSM noise to the input image.
    Returns dict containing the adversarial image path and new prediction info.
    """
    try:
        model = registry.get(model_name)
        if model is None:
            return None

        h, w, c = _model_input_shape(model)
        tensor = tf.Variable(preprocess_image(image_array, h, w, c), dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            preds = model(tensor)
            orig_class = tf.argmax(preds, axis=-1)
            loss = tf.keras.losses.sparse_categorical_crossentropy(orig_class, preds)
        
        gradient = tape.gradient(loss, tensor)
        signed_grad = tf.sign(gradient)
        
        # Create adversarial example
        adv_x = tensor + (epsilon * signed_grad)
        
        if c == 1:
            adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)
        else:
            adv_x = tf.clip_by_value(adv_x, -1.0, 1.0)
            
        adv_preds = model.predict(adv_x, verbose=0)[0]
        new_class = int(np.argmax(adv_preds))
        new_conf = float(adv_preds[new_class])
        
        adv_x_np = adv_x.numpy()[0]
        if c == 3:
            adv_img = np.uint8((adv_x_np + 1.0) * 127.5)
        else:
            adv_grey = (adv_x_np[:, :, 0] * 255).astype(np.uint8)
            adv_img = cv2.cvtColor(adv_grey, cv2.COLOR_GRAY2BGR)
            
        fname = f"adv_{uuid.uuid4().hex[:10]}.png"
        out_path = UPLOADS_DIR / fname
        cv2.imwrite(str(out_path), adv_img)
        
        return {
            "adversarial_image_path": f"static/uploads/{fname}",
            "original_digit": int(orig_class[0]),
            "predicted_digit": new_class,
            "confidence": round(new_conf * 100, 2)
        }
        
    except Exception as exc:
        logger.error(f"FGSM failed: {exc}")
        raise


# ── Worksheet-only digit prediction helper ───────────────────────────────────
def predict_worksheet_digits_only(
    pil_img,
    model_name: str = "cnn_deep",
    confidence_threshold: float = 70.0,
) -> Dict[str, Any]:
    """
    Detect digits from worksheet-style images and format them row-wise.

    Uses the existing lazy model loader:
        model = registry.get(model_name)

    Parameters
    ----------
    pil_img:
        PIL Image or RGB numpy image.
    model_name:
        Name of model inside models/ folder, for example "cnn_deep" or "cnn_medium".
    confidence_threshold:
        You can pass 70 / 0.70. This function converts it safely for the
        worksheet detector.

    Returns
    -------
    A Streamlit-compatible result dictionary.
    """
    try:
        from preprocessing.multidigit import detect_worksheet_digits_only
    except Exception as exc:
        raise ImportError(
            "detect_worksheet_digits_only was not found in preprocessing/multidigit.py. "
            "Add that function there first, or use multi_digit_predict fallback."
        ) from exc

    model = registry.get(model_name)
    if model is None:
        raise ValueError(f"Model '{model_name}' not available. Train it first.")

    # Accept PIL image or numpy RGB image from Streamlit.
    if isinstance(pil_img, np.ndarray):
        arr = pil_img.astype(np.uint8)
        if arr.ndim == 2:
            pil_img = Image.fromarray(arr).convert("RGB")
        elif arr.ndim == 3 and arr.shape[2] == 4:
            pil_img = Image.fromarray(arr[:, :, :3]).convert("RGB")
        else:
            pil_img = Image.fromarray(arr).convert("RGB")
    else:
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")

    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # User UI often gives 30/50/70, detector may expect 0.30/0.50/0.70.
    thr = float(confidence_threshold)
    if thr > 1.0:
        thr = thr / 100.0

    result = detect_worksheet_digits_only(
        img_bgr,
        model=model,
        confidence_threshold=thr,
    )

    rows_raw = result.get("rows", []) if isinstance(result, dict) else []
    formatted_rows = []
    flat_digits = []

    for row_index, row in enumerate(rows_raw, start=1):
        row_digits = []
        for col_index, item in enumerate(row, start=1):
            digit = item.get("digit", item.get("predicted_digit", ""))
            try:
                digit = int(digit)
            except Exception:
                pass

            conf = item.get("confidence", item.get("score", 0.0))
            try:
                conf = float(conf)
                if conf <= 1.0:
                    conf *= 100.0
            except Exception:
                conf = 0.0

            box = item.get("box", item.get("bbox", item.get("rect", "")))

            row_digits.append(digit)
            flat_digits.append({
                "position": len(flat_digits) + 1,
                "digit": digit,
                "confidence": round(conf, 2),
                "status": "ok" if conf >= (thr * 100.0) else "low_confidence",
                "row": row_index,
                "column": col_index,
                "box": box,
                "top3_predictions": item.get("top3_predictions", []),
                "canvas28": item.get("canvas28", None),
            })

        # Format row-wise output as pairs: [9,4] [9,2] ...
        pairs = []
        i = 0
        while i < len(row_digits):
            if i + 1 < len(row_digits):
                pairs.append([row_digits[i], row_digits[i + 1]])
                i += 2
            else:
                pairs.append([row_digits[i]])
                i += 1
        formatted_rows.append(pairs)

    prediction = "".join(str(d["digit"]) for d in flat_digits)
    low_count = sum(1 for d in flat_digits if d["status"] != "ok")

    return {
        "success": len(flat_digits) > 0,
        "mode": "worksheet_digits_only",
        "prediction": prediction,
        "raw_prediction": prediction,
        "digit_count": len(flat_digits),
        "uncertain_count": low_count,
        "message": "OK" if flat_digits else "No worksheet digits detected.",
        "formatted_rows": formatted_rows,
        "rows": rows_raw,
        "digits": flat_digits,
        "visualized": result.get("visualized") if isinstance(result, dict) else None,
        "annotated_image": result.get("visualized") if isinstance(result, dict) else None,
        "binary": result.get("binary") if isinstance(result, dict) else None,
        "mask_image": result.get("binary") if isinstance(result, dict) else None,
        "enhanced_gray": result.get("enhanced_gray") if isinstance(result, dict) else None,
        "model_used": model_name,
    }


def detect_digits_auto(
    image_array: np.ndarray,
    model_name: str = "cnn_deep",
    confidence_threshold: float = 50.0,
    max_digits: int = 50,
    use_tta: bool = True,
    min_margin: float = 4.0,
    rotation_angle: float = 0.0,
    auto_rotate: bool = False,
    auto_deskew: bool = True,
    auto_crop: bool = False,
    try_worksheet: bool = True,
) -> Dict[str, Any]:
    """
    Auto detector used by Streamlit.

    First tries worksheet-specific detector when try_worksheet=True.
    If unavailable or failed, it falls back to normal multi_digit_predict().
    Extra arguments are accepted so Streamlit does not crash with unexpected
    keyword argument errors.
    """
    if try_worksheet:
        try:
            worksheet_result = predict_worksheet_digits_only(
                image_array,
                model_name=model_name,
                confidence_threshold=confidence_threshold,
            )
            if worksheet_result.get("success"):
                return worksheet_result
        except Exception as exc:
            logger.warning(f"Worksheet detector fallback: {exc}")

    # Fallback to your existing general multi-digit detector.
    return multi_digit_predict(
        image_array,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        max_digits=max_digits,
    )

