"""
Robust OCR prediction service for the DigitAI project.

Replace:
    backend/services/ocr_prediction_service.py

What this fixes:
    1) Full images are no longer blindly compressed into one CRNN input.
    2) Alphabet charts / A-Z grids are handled using character-grid segmentation.
    3) Word/script images are handled using line segmentation.
    4) charset (4).json is accepted as a fallback if charset.json is missing.
    5) CTC training models are converted to inference models automatically.

Important:
    Better preprocessing can improve results, but it cannot make an under-trained
    model read every font/script perfectly. For high accuracy on your sample images,
    retrain OCR using training/train_synthetic_ocr_fonts_kaggle.py from the fix pack.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageOps, ImageFilter

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

PREDICTION_MODEL_PATH = MODELS_DIR / "ocr_prediction_model.keras"
TRAINING_MODEL_PATH = MODELS_DIR / "best_ocr_training_model.keras"
CHARSET_PATH = MODELS_DIR / "charset.json"

DEFAULT_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,;:!?'-/()&"

Box = Tuple[int, int, int, int]


@dataclass
class OCRBox:
    box: Box
    row: int = 0


# ==========================================================
# Model loading
# ==========================================================

@tf.keras.utils.register_keras_serializable(name="ctc_loss_func")
def _ctc_loss_func(args):
    labels_true, pred, input_length, label_length = args
    return K.ctc_batch_cost(labels_true, pred, input_length, label_length)


def _find_charset_file() -> Path:
    if CHARSET_PATH.exists():
        return CHARSET_PATH
    matches = sorted(MODELS_DIR.glob("charset*.json"))
    if matches:
        return matches[0]
    return CHARSET_PATH


def load_charset_config(charset_path: Union[str, Path, None] = None) -> Dict[str, Any]:
    path = Path(charset_path) if charset_path else _find_charset_file()
    cfg: Dict[str, Any] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    charset = str(cfg.get("charset", DEFAULT_CHARSET))
    return {
        "charset": charset,
        "blank_index": int(cfg.get("blank_index", len(charset))),
        "img_h": int(cfg.get("img_h", 32)),
        "img_w": int(cfg.get("img_w", 384)),
        "max_label_len": int(cfg.get("max_label_len", 64)),
        "num_classes": int(cfg.get("num_classes", len(charset) + 1)),
        "charset_path": str(path),
    }


def _extract_pred_model_from_training(m: tf.keras.Model) -> tf.keras.Model:
    """Extract image -> softmax inference model from a CTC training model."""
    ctc_out_tensor = None
    for name_candidate in ("ctc_softmax", "softmax", "dense"):
        try:
            ctc_out_tensor = m.get_layer(name_candidate).output
            break
        except Exception:
            pass
    if ctc_out_tensor is None:
        for layer in reversed(m.layers):
            if "softmax" in layer.name.lower():
                ctc_out_tensor = layer.output
                break
    if ctc_out_tensor is None:
        # Some exported models already end with predictions.
        ctc_out_tensor = m.output

    aux_names = {"label", "labels", "input_length", "label_length"}
    input_layers = [l for l in m.layers if isinstance(l, tf.keras.layers.InputLayer)]

    candidates = []
    for l in input_layers:
        if l.name not in aux_names:
            candidates.append(l)
    try:
        explicit = m.get_layer("image")
        if explicit not in candidates:
            candidates.insert(0, explicit)
    except Exception:
        pass

    for inp_layer in candidates + input_layers:
        try:
            return models.Model(inputs=inp_layer.output, outputs=ctc_out_tensor, name="ocr_prediction_model")
        except Exception:
            continue

    raise ValueError(
        "Could not extract OCR prediction model. Export an inference model as models/ocr_prediction_model.keras."
    )


def _load_keras_model(path: Path) -> tf.keras.Model:
    m = tf.keras.models.load_model(
        str(path),
        compile=False,
        safe_mode=False,
        custom_objects={"ctc_loss_func": _ctc_loss_func},
    )
    try:
        if len(m.inputs) > 1:
            m = _extract_pred_model_from_training(m)
    except Exception:
        pass
    return m


@lru_cache(maxsize=1)
def load_ocr_model() -> tf.keras.Model:
    if PREDICTION_MODEL_PATH.exists():
        return _load_keras_model(PREDICTION_MODEL_PATH)
    if TRAINING_MODEL_PATH.exists():
        return _load_keras_model(TRAINING_MODEL_PATH)
    raise FileNotFoundError(
        "OCR model not found. Add models/ocr_prediction_model.keras or models/best_ocr_training_model.keras"
    )


@lru_cache(maxsize=8)
def load_ocr_model_from_path(model_path: str) -> tf.keras.Model:
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"OCR model not found: {p}")
    return _load_keras_model(p)


def get_available_ocr_models() -> Dict[str, Path]:
    """Return only OCR/CTC models. Digit CNN files are intentionally ignored."""
    candidates: Dict[str, Path] = {}
    for p in sorted(MODELS_DIR.glob("*.keras")):
        stem = p.stem.lower()
        if any(k in stem for k in ("ocr", "crnn", "ctc")):
            candidates[p.stem.replace("_", " ").title()] = p
    if PREDICTION_MODEL_PATH.exists():
        candidates.setdefault(PREDICTION_MODEL_PATH.stem.replace("_", " ").title(), PREDICTION_MODEL_PATH)
    return candidates


# ==========================================================
# Image helpers
# ==========================================================


def to_pil_image(image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image)
    if isinstance(image, (str, Path)):
        return ImageOps.exif_transpose(Image.open(image))
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr)
        if arr.ndim == 3 and arr.shape[2] == 4:
            return Image.fromarray(arr, mode="RGBA").convert("RGB")
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return Image.fromarray(arr[:, :, :3])
    raise TypeError(f"Unsupported image type: {type(image)}")


def auto_make_dark_text_on_white(gray_img: Image.Image) -> Image.Image:
    img = gray_img.convert("L")
    arr = np.asarray(img)
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if float(np.median(border)) < 127:
        img = ImageOps.invert(img)
    return img


def crop_content(gray_img: Image.Image, padding: int = 2) -> Image.Image:
    img = gray_img.convert("L")
    arr = np.asarray(img)
    # Works for black text on white background.
    mask = arr < 245
    ys, xs = np.where(mask)
    if len(xs) < 5 or len(ys) < 5:
        return img
    left = max(0, int(xs.min()) - padding)
    right = min(img.width, int(xs.max()) + 1 + padding)
    top = max(0, int(ys.min()) - padding)
    bottom = min(img.height, int(ys.max()) + 1 + padding)
    return img.crop((left, top, right, bottom))


def preprocess_script_line_image(
    image: Union[str, Path, Image.Image, np.ndarray],
    img_h: int = 32,
    img_w: int = 384,
    crop: bool = True,
) -> np.ndarray:
    img = to_pil_image(image).convert("L")
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = auto_make_dark_text_on_white(img)
    if crop:
        img = crop_content(img, padding=3)

    w, h = img.size
    scale = min(img_w / max(w, 1), img_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (img_w, img_h), 255)
    canvas.paste(img, (0, (img_h - new_h) // 2))
    arr = np.asarray(canvas).astype("float32")
    arr = (255.0 - arr) / 255.0
    return arr[..., None].astype("float32")


# ==========================================================
# Segmentation
# ==========================================================


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError("opencv-python is required. Run: pip install opencv-python")


def _text_mask(image: Image.Image) -> np.ndarray:
    """Return foreground mask where text/strokes are 255 and background is 0."""
    _require_cv2()
    rgb = np.asarray(image.convert("RGB"))
    gray = np.asarray(auto_make_dark_text_on_white(image.convert("L")))

    # Shadow / textured background correction.
    h, w = gray.shape[:2]
    k = max(31, int(min(h, w) * 0.08) | 1)
    bg = cv2.GaussianBlur(gray, (k, k), 0)
    corrected = cv2.divide(gray, bg, scale=255)
    corrected = cv2.GaussianBlur(corrected, (3, 3), 0)

    _, otsu = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    block = max(31, int(min(h, w) * 0.045) | 1)
    if block >= min(h, w):
        block = max(3, (min(h, w) // 2) * 2 - 1)
    adaptive = cv2.adaptiveThreshold(
        corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 11
    )

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    # Catches red numbers / colored letters that may be light in grayscale.
    color_mask = ((sat > 28) & (val > 35)).astype(np.uint8) * 255

    mask = cv2.bitwise_or(otsu, adaptive)
    mask = cv2.bitwise_or(mask, color_mask)

    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)

    border = max(1, int(min(h, w) * 0.003))
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0
    return mask


def _sort_rows(boxes: Sequence[Box], y_tol: Optional[float] = None) -> List[List[Box]]:
    boxes = list(boxes)
    if not boxes:
        return []
    boxes.sort(key=lambda b: (b[1] + b[3] / 2.0, b[0]))
    med_h = float(np.median([b[3] for b in boxes]))
    tol = y_tol if y_tol is not None else max(10.0, med_h * 0.60)

    rows: List[List[Box]] = []
    for b in boxes:
        cy = b[1] + b[3] / 2.0
        placed = False
        for row in rows:
            row_cy = float(np.mean([r[1] + r[3] / 2.0 for r in row]))
            if abs(cy - row_cy) <= tol:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])

    rows.sort(key=lambda row: float(np.mean([r[1] + r[3] / 2.0 for r in row])))
    for row in rows:
        row.sort(key=lambda b: b[0])
    return rows


def _dedupe_boxes(boxes: Iterable[Box], iou_threshold: float = 0.80) -> List[Box]:
    def iou(a: Box, b: Box) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    out: List[Box] = []
    for b in sorted(boxes, key=lambda x: x[2] * x[3], reverse=True):
        if all(iou(b, k) < iou_threshold for k in out):
            out.append(b)
    return sorted(out, key=lambda b: (b[1], b[0]))


def detect_text_line_boxes(image: Union[str, Path, Image.Image, np.ndarray]) -> List[Box]:
    img = to_pil_image(image).convert("RGB")
    W, H = img.size
    mask = _text_mask(img)

    # Join letters into words/lines. Smaller kernel prevents merging separate vertical lines.
    kw = max(14, int(W * 0.035))
    kh = max(3, int(H * 0.012))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    dil = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Box] = []
    min_area = max(40, int(W * H * 0.0007))
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < min_area:
            continue
        if h < max(8, int(H * 0.025)) or w < max(8, int(W * 0.015)):
            continue
        if area > W * H * 0.95:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))

    rows = _sort_rows(_dedupe_boxes(boxes, 0.75))
    merged: List[Box] = []
    # If a row has multiple word boxes, merge them into one line box.
    for row in rows:
        x1 = min(b[0] for b in row)
        y1 = min(b[1] for b in row)
        x2 = max(b[0] + b[2] for b in row)
        y2 = max(b[1] + b[3] for b in row)
        merged.append((x1, y1, x2 - x1, y2 - y1))

    # Final noise filter: textured paper/shadows can create thin fake line boxes.
    # Keep boxes that contain enough real foreground pixels.
    filtered: List[Box] = []
    image_area = float(W * H)
    for x, y, w, h in merged:
        roi = mask[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
        ink = int(cv2.countNonZero(roi)) if roi.size else 0
        box_area = max(1, int(w * h))
        density = ink / float(box_area)
        if ink < max(35, int(image_area * 0.00012)):
            continue
        if density < 0.010:
            continue
        filtered.append((int(x), int(y), int(w), int(h)))

    return filtered


def _merge_dot_boxes(boxes: List[Box]) -> List[Box]:
    """Merge dot-like components above i/j style characters."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3])
    used = [False] * len(boxes)
    merged: List[Box] = []

    for i, small in enumerate(boxes):
        if used[i]:
            continue
        sx, sy, sw, sh = small
        sarea = sw * sh
        did_merge = False
        if sarea <= 1500:
            for j, big in enumerate(boxes):
                if i == j or used[j]:
                    continue
                bx, by, bw, bh = big
                barea = bw * bh
                if barea <= sarea * 1.6:
                    continue
                center_inside_x = (bx - bw * 0.25) <= (sx + sw / 2) <= (bx + bw * 1.25)
                above = sy + sh <= by + bh * 0.35
                gap = by - (sy + sh)
                close = gap <= max(12, int(bh * 0.55))
                if center_inside_x and above and close:
                    x1, y1 = min(sx, bx), min(sy, by)
                    x2, y2 = max(sx + sw, bx + bw), max(sy + sh, by + bh)
                    merged.append((x1, y1, x2 - x1, y2 - y1))
                    used[i] = used[j] = True
                    did_merge = True
                    break
        if not did_merge and not used[i]:
            used[i] = True
            merged.append(small)
    return _dedupe_boxes(merged, 0.70)


def detect_character_boxes(image: Union[str, Path, Image.Image, np.ndarray]) -> List[Box]:
    img = to_pil_image(image).convert("RGB")
    W, H = img.size
    mask = _text_mask(img)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: List[Box] = []
    image_area = W * H
    min_area = max(8, int(image_area * 0.000025))
    max_area = int(image_area * 0.25)

    for i in range(1, num_labels):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if area < min_area or area > max_area:
            continue
        if w < 2 or h < 4:
            continue
        if h > H * 0.85 or w > W * 0.85:
            continue
        aspect = w / float(max(h, 1))
        if aspect > 4.0 or aspect < 0.02:
            continue
        boxes.append((x, y, w, h))

    boxes = _merge_dot_boxes(boxes)
    rows = _sort_rows(boxes)
    return [b for row in rows for b in row]


def crop_boxes(image: Union[str, Path, Image.Image, np.ndarray], boxes: Sequence[Box], padding: int = 6) -> List[Image.Image]:
    img = to_pil_image(image).convert("RGB")
    W, H = img.size
    crops = []
    for x, y, w, h in boxes:
        left = max(0, x - padding)
        top = max(0, y - padding)
        right = min(W, x + w + padding)
        bottom = min(H, y + h + padding)
        crops.append(img.crop((left, top, right, bottom)))
    return crops


def _looks_like_character_grid(image: Image.Image) -> bool:
    boxes = detect_character_boxes(image)
    if len(boxes) < 8:
        return False
    rows = _sort_rows(boxes)
    if len(rows) >= 2 and max(len(r) for r in rows) >= 4:
        return True
    # Large separated title-like letters also benefit from character mode.
    if len(rows) <= 3:
        widths = [b[2] for b in boxes]
        heights = [b[3] for b in boxes]
        if np.median(heights) > image.height * 0.12 and len(boxes) >= 6:
            return True
    return False


# ==========================================================
# CTC decoding
# ==========================================================


def decode_ctc_predictions_with_confidences(
    pred: np.ndarray,
    charset: str,
    blank_index: Optional[int] = None,
) -> Tuple[List[str], List[List[Dict[str, Any]]]]:
    if blank_index is None:
        blank_index = len(charset)
    num_to_char = {i: ch for i, ch in enumerate(charset)}

    results: List[str] = []
    details_all: List[List[Dict[str, Any]]] = []

    for i in range(pred.shape[0]):
        max_probs = np.max(pred[i], axis=-1)
        best_ids = np.argmax(pred[i], axis=-1)
        text = ""
        details: List[Dict[str, Any]] = []
        prev_id = -1
        for t, cid in enumerate(best_ids):
            cid_int = int(cid)
            if cid_int == blank_index or cid_int == -1:
                prev_id = cid_int
                continue
            if cid_int != prev_id:
                ch = num_to_char.get(cid_int, "")
                if ch:
                    text += ch
                    details.append({"char": ch, "confidence": float(max_probs[t])})
            elif details:
                details[-1]["confidence"] = max(details[-1]["confidence"], float(max_probs[t]))
            prev_id = cid_int
        results.append(text)
        details_all.append(details)
    return results, details_all


def _confidence_from_pred(pred: np.ndarray, blank_index: int) -> float:
    max_probs = np.max(pred[0], axis=-1)
    best_ids = np.argmax(pred[0], axis=-1)
    non_blank = best_ids != blank_index
    if np.any(non_blank):
        return float(np.mean(max_probs[non_blank]))
    return float(np.mean(max_probs))


def predict_ocr_crop(
    image: Union[str, Path, Image.Image, np.ndarray],
    model: Optional[tf.keras.Model] = None,
) -> Dict[str, Any]:
    cfg = load_charset_config()
    charset = cfg["charset"]
    if model is None:
        model = load_ocr_model()

    arr = preprocess_script_line_image(image, img_h=cfg["img_h"], img_w=cfg["img_w"], crop=True)
    pred = model.predict(np.expand_dims(arr, axis=0), verbose=0)
    texts, details = decode_ctc_predictions_with_confidences(pred, charset, cfg["blank_index"])
    return {
        "text": texts[0],
        "confidence": _confidence_from_pred(pred, cfg["blank_index"]),
        "char_details": details[0],
        "model_type": "ocr_crnn_ctc_crop",
    }


def _first_alnum(text: str) -> str:
    # Prefer alphanumeric output for single-character grid cells.
    for ch in text:
        if ch.isalnum():
            return ch
    return text[:1] if text else ""


def predict_character_grid_ocr(
    image: Union[str, Path, Image.Image, np.ndarray],
    model: Optional[tf.keras.Model] = None,
) -> Dict[str, Any]:
    img = to_pil_image(image).convert("RGB")
    if model is None:
        model = load_ocr_model()
    boxes = detect_character_boxes(img)
    rows = _sort_rows(boxes)

    line_results: List[Dict[str, Any]] = []
    all_conf: List[float] = []

    for row_idx, row in enumerate(rows):
        chars: List[str] = []
        char_details: List[Dict[str, Any]] = []
        # Space detection using horizontal gaps.
        row_heights = [b[3] for b in row] or [1]
        gap_threshold = max(10, int(np.median(row_heights) * 0.65))
        prev_right = None
        for b in row:
            if prev_right is not None and b[0] - prev_right > gap_threshold:
                chars.append(" ")
            crop = crop_boxes(img, [b], padding=4)[0]
            r = predict_ocr_crop(crop, model=model)
            ch = _first_alnum(r.get("text", ""))
            conf = float(r.get("confidence", 0.0) or 0.0)
            chars.append(ch)
            all_conf.append(conf)
            char_details.append({
                "char": ch,
                "confidence": conf,
                "box": b,
                "raw_text": r.get("text", ""),
            })
            prev_right = b[0] + b[2]
        line_text = "".join(chars).strip()
        line_results.append({
            "text": line_text,
            "confidence": float(np.mean([d["confidence"] for d in char_details])) if char_details else None,
            "char_details": char_details,
            "line_index": row_idx,
        })

    text = "\n".join([r["text"] for r in line_results if r.get("text")])
    return {
        "text": text,
        "confidence": float(np.mean(all_conf)) if all_conf else None,
        "lines": line_results,
        "boxes": boxes,
        "used_segmentation": True,
        "segmentation_mode": "character_grid",
    }



# ==========================================================
# Line OCR noise filtering
# ==========================================================

def _clean_line_text(text: str) -> str:
    """Normalize OCR line text for display/filtering without changing real characters."""
    text = str(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _trim_edge_low_conf_chars(result: Dict[str, Any], edge_conf: float = 0.42) -> Dict[str, Any]:
    """
    CTC can sometimes add a very weak first/last character from paper texture or
    border strokes. Trim only weak edge chars; do not touch middle characters.
    """
    details = list(result.get("char_details") or [])
    if len(details) <= 2:
        result["text"] = _clean_line_text(result.get("text", ""))
        return result

    changed = False
    while len(details) > 2 and float(details[0].get("confidence", 0.0) or 0.0) < edge_conf:
        details.pop(0)
        changed = True
    while len(details) > 2 and float(details[-1].get("confidence", 0.0) or 0.0) < edge_conf:
        details.pop()
        changed = True

    if changed:
        result["char_details"] = details
        result["text"] = "".join(str(d.get("char", "")) for d in details).strip()
        result["confidence"] = float(np.mean([float(d.get("confidence", 0.0) or 0.0) for d in details])) if details else None
    else:
        result["text"] = _clean_line_text(result.get("text", ""))
    return result


def _should_keep_line_result(result: Dict[str, Any]) -> bool:
    """
    Remove false OCR lines created by background texture/noise.

    For word/line mode, short predictions like 'S', 'SS', 'CS' with low confidence
    are almost always noise when the uploaded image contains full words.
    Use Character Grid or Whole Image mode when you intentionally want one-letter OCR.
    """
    text = _clean_line_text(result.get("text", ""))
    conf = float(result.get("confidence", 0.0) or 0.0)
    alnum_count = sum(1 for ch in text if ch.isalnum())

    if alnum_count == 0:
        return False

    # Very weak line = noise/background.
    if conf < 0.30:
        return False

    # In line/word mode, discard low-confidence tiny fragments.
    if alnum_count <= 1 and conf < 0.80:
        return False
    if alnum_count <= 2 and conf < 0.65:
        return False

    return True


def _filter_line_ocr_results(line_results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept: List[Dict[str, Any]] = []
    discarded: List[Dict[str, Any]] = []

    for r in line_results:
        r = _trim_edge_low_conf_chars(r)
        r["text"] = _clean_line_text(r.get("text", ""))
        if _should_keep_line_result(r):
            kept.append(r)
        else:
            discarded.append(r)

    # Safety fallback: if everything was filtered, keep the best non-empty line.
    if not kept and line_results:
        non_empty = [r for r in line_results if _clean_line_text(r.get("text", ""))]
        if non_empty:
            best = max(non_empty, key=lambda x: float(x.get("confidence", 0.0) or 0.0))
            best["text"] = _clean_line_text(best.get("text", ""))
            kept = [best]
            discarded = [r for r in line_results if r is not best]

    for i, r in enumerate(kept):
        r["line_index"] = i
    return kept, discarded


def predict_line_ocr(
    image: Union[str, Path, Image.Image, np.ndarray],
    model: Optional[tf.keras.Model] = None,
    join_lines_with: str = "\n",
) -> Dict[str, Any]:
    img = to_pil_image(image).convert("RGB")
    if model is None:
        model = load_ocr_model()
    boxes = detect_text_line_boxes(img)

    if not boxes:
        one = predict_ocr_crop(img, model=model)
        one = _trim_edge_low_conf_chars(one)
        one["line_index"] = 0
        kept = [one] if _should_keep_line_result(one) else []
        return {
            "text": one["text"] if kept else "",
            "confidence": one["confidence"] if kept else None,
            "lines": kept,
            "raw_lines": [one],
            "discarded_lines": [] if kept else [one],
            "boxes": [],
            "used_segmentation": False,
            "segmentation_mode": "whole_image",
        }

    raw_line_results = []
    for idx, (box, crop) in enumerate(zip(boxes, crop_boxes(img, boxes, padding=8))):
        r = predict_ocr_crop(crop, model=model)
        r["box"] = box
        r["line_index"] = idx
        raw_line_results.append(r)

    line_results, discarded_lines = _filter_line_ocr_results(raw_line_results)

    texts = [r["text"] for r in line_results if r.get("text")]
    confs = [float(r["confidence"]) for r in line_results if r.get("confidence") is not None]
    return {
        "text": join_lines_with.join(texts),
        "confidence": float(np.mean(confs)) if confs else None,
        "lines": line_results,
        "raw_lines": raw_line_results,
        "discarded_lines": discarded_lines,
        "boxes": [r.get("box") for r in line_results if r.get("box")],
        "used_segmentation": True,
        "segmentation_mode": "line",
    }


# ==========================================================
# Printed paragraph OCR fallback
# ==========================================================

@lru_cache(maxsize=1)
def _get_easyocr_reader():
    """Lazy EasyOCR reader. Requires: pip install easyocr"""
    import easyocr
    return easyocr.Reader(["en"], gpu=False)


def _easyocr_box_to_xywh(box) -> Box:
    pts = np.asarray(box, dtype=np.float32)
    x1 = int(np.floor(np.min(pts[:, 0])))
    y1 = int(np.floor(np.min(pts[:, 1])))
    x2 = int(np.ceil(np.max(pts[:, 0])))
    y2 = int(np.ceil(np.max(pts[:, 1])))
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def _group_ocr_words_into_lines(items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not items:
        return []
    items = sorted(items, key=lambda r: (r["box"][1], r["box"][0]))
    heights = [r["box"][3] for r in items]
    med_h = float(np.median(heights)) if heights else 12.0
    y_tol = max(8.0, med_h * 0.65)
    rows: List[List[Dict[str, Any]]] = []
    for item in items:
        x, y, w, h = item["box"]
        cy = y + h / 2.0
        placed = False
        for row in rows:
            row_cy = np.mean([r["box"][1] + r["box"][3] / 2.0 for r in row])
            if abs(cy - row_cy) <= y_tol:
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])
    for row in rows:
        row.sort(key=lambda r: r["box"][0])
    rows.sort(key=lambda row: np.mean([r["box"][1] for r in row]))
    return rows


def predict_printed_paragraph_ocr(
    image: Union[str, Path, Image.Image, np.ndarray],
    min_confidence: float = 0.20,
) -> Dict[str, Any]:
    """
    OCR for printed paragraphs/screenshots using EasyOCR/Tesseract fallback.
    This is for small printed document text. It does not use the custom CRNN model.
    """
    img = to_pil_image(image).convert("RGB")
    arr = np.asarray(img)
    easyocr_error = "not installed"

    try:
        reader = _get_easyocr_reader()
        raw = reader.readtext(arr, detail=1, paragraph=False)
        words: List[Dict[str, Any]] = []
        for box, text, conf in raw:
            text = str(text).strip()
            conf = float(conf)
            if not text or conf < min_confidence:
                continue
            words.append({
                "text": text,
                "confidence": conf,
                "box": _easyocr_box_to_xywh(box),
                "engine": "easyocr",
            })
        rows = _group_ocr_words_into_lines(words)
        line_results: List[Dict[str, Any]] = []
        for i, row in enumerate(rows):
            line_text = " ".join(r["text"] for r in row).strip()
            if not line_text:
                continue
            conf = float(np.mean([r["confidence"] for r in row])) if row else None
            x1 = min(r["box"][0] for r in row)
            y1 = min(r["box"][1] for r in row)
            x2 = max(r["box"][0] + r["box"][2] for r in row)
            y2 = max(r["box"][1] + r["box"][3] for r in row)
            line_results.append({
                "line_index": i,
                "text": line_text,
                "confidence": conf,
                "box": (x1, y1, x2 - x1, y2 - y1),
                "char_details": [{"char": ch, "confidence": conf} for ch in line_text],
                "words": row,
            })
        confs = [r["confidence"] for r in line_results if r.get("confidence") is not None]
        return {
            "text": "\n".join(r["text"] for r in line_results),
            "confidence": float(np.mean(confs)) if confs else None,
            "lines": line_results,
            "raw_lines": line_results,
            "discarded_lines": [],
            "boxes": [r.get("box") for r in line_results if r.get("box")],
            "used_segmentation": True,
            "segmentation_mode": "printed_paragraph_easyocr",
        }
    except ModuleNotFoundError:
        pass
    except Exception as e:
        easyocr_error = str(e)

    try:
        import pytesseract
        from pytesseract import Output
        data = pytesseract.image_to_data(img, output_type=Output.DICT, config="--oem 3 --psm 6")
        words: List[Dict[str, Any]] = []
        n = len(data.get("text", []))
        for i in range(n):
            text = str(data["text"][i]).strip()
            try:
                conf = float(data["conf"][i])
            except Exception:
                conf = -1.0
            if not text or conf < 20:
                continue
            words.append({
                "text": text,
                "confidence": conf / 100.0,
                "box": (int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])),
                "engine": "pytesseract",
            })
        rows = _group_ocr_words_into_lines(words)
        line_results: List[Dict[str, Any]] = []
        for i, row in enumerate(rows):
            line_text = " ".join(r["text"] for r in row).strip()
            if not line_text:
                continue
            conf = float(np.mean([r["confidence"] for r in row])) if row else None
            line_results.append({
                "line_index": i,
                "text": line_text,
                "confidence": conf,
                "box": None,
                "char_details": [{"char": ch, "confidence": conf} for ch in line_text],
                "words": row,
            })
        confs = [r["confidence"] for r in line_results if r.get("confidence") is not None]
        return {
            "text": "\n".join(r["text"] for r in line_results),
            "confidence": float(np.mean(confs)) if confs else None,
            "lines": line_results,
            "raw_lines": line_results,
            "discarded_lines": [],
            "boxes": [r.get("box") for r in line_results if r.get("box")],
            "used_segmentation": True,
            "segmentation_mode": "printed_paragraph_tesseract",
        }
    except Exception as tess_err:
        raise RuntimeError(
            "Printed Paragraph mode needs EasyOCR or Tesseract. Install one of these:\n"
            "1) pip install easyocr\n"
            "or\n"
            "2) install Tesseract OCR for Windows and run: pip install pytesseract\n"
            f"EasyOCR error: {easyocr_error}\n"
            f"Tesseract error: {tess_err}"
        )


def predict_full_image_ocr(
    image: Union[str, Path, Image.Image, np.ndarray],
    model: Optional[tf.keras.Model] = None,
    join_lines_with: str = "\n",
    mode: str = "auto",
    debug: bool = False,
) -> Dict[str, Any]:
    """
    mode:
        auto              -> choose character grid or line OCR automatically
        character_grid    -> best for A-Z charts, letter/number tables, separated letters
        line              -> best for short words/script words using the custom CRNN model
        printed_paragraph -> best for small printed paragraphs/screenshots using EasyOCR/Tesseract
        whole             -> force one crop through OCR model
    """
    img = to_pil_image(image).convert("RGB")
    mode = (mode or "auto").lower().replace("-", "_").replace(" ", "_")

    if mode in {"printed", "printed_paragraph", "paragraph", "document", "screenshot_text"}:
        out = predict_printed_paragraph_ocr(img)
    else:
        if model is None:
            model = load_ocr_model()

        if mode in {"character", "characters", "char", "char_grid", "character_grid", "grid"}:
            out = predict_character_grid_ocr(img, model=model)
        elif mode in {"line", "lines", "word", "words", "text"}:
            out = predict_line_ocr(img, model=model, join_lines_with=join_lines_with)
        elif mode in {"whole", "crop", "single"}:
            one = predict_ocr_crop(img, model=model)
            one["line_index"] = 0
            out = {
                "text": one["text"],
                "confidence": one["confidence"],
                "lines": [one],
                "boxes": [],
                "used_segmentation": False,
                "segmentation_mode": "whole_image",
            }
        else:
            try:
                use_grid = _looks_like_character_grid(img)
            except Exception:
                use_grid = False
            if use_grid:
                out = predict_character_grid_ocr(img, model=model)
            else:
                out = predict_line_ocr(img, model=model, join_lines_with=join_lines_with)

    if debug:
        try:
            out["debug_char_box_count"] = len(detect_character_boxes(img))
            out["debug_line_box_count"] = len(detect_text_line_boxes(img))
        except Exception:
            pass
    return out

def predict_ocr_text(
    image: Union[str, Path, Image.Image, np.ndarray],
    model: Optional[tf.keras.Model] = None,
    auto_segment: bool = True,
) -> Dict[str, Any]:
    return predict_full_image_ocr(image, model=model, mode="auto" if auto_segment else "whole")


if __name__ == "__main__":
    print("Loading OCR model...")
    m = load_ocr_model()
    print("Loaded:", m.name)
    print("Input:", m.input_shape)
    print("Output:", m.output_shape)
    print("Charset:", load_charset_config()["charset_path"])
