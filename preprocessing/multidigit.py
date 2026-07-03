"""
Universal multi-digit preprocessing and segmentation.

Put this file at:
    preprocessing/multidigit.py

Goal:
    Detect handwritten / printed digits from many image types:
    - black pen on white paper
    - white digit on black/dark background
    - red/pink faded marker
    - blue/green/purple colored ink
    - pencil / low contrast writing
    - phone camera images with shadows/noise
    - scanned images
    - multi-line / multi-digit images

Output format is compatible with backend/services/prediction_service.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class DigitCandidate:
    box: Box
    canvas28: np.ndarray  # 28x28 uint8, black background, white digit
    crop: np.ndarray      # binary crop


def auto_orient_for_multidigit_line(image: np.ndarray) -> np.ndarray:
    """Rotate very tall phone images into a normal horizontal digit-line view."""
    arr = np.asarray(image)
    if arr.ndim >= 2:
        h, w = arr.shape[:2]
        if h > w * 1.45:
            return cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
    return arr


def _to_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGBA2RGB)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, :3].astype(np.uint8)
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def _to_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3 and arr.shape[2] == 4:
        gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        # This works acceptably for RGB and BGR because grayscale weights differ only slightly.
        gray = cv2.cvtColor(arr[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def _clean_mask(mask: np.ndarray, image_shape: Tuple[int, int], connect: bool = True) -> np.ndarray:
    h, w = image_shape
    mask = (mask > 0).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    if connect:
        k = max(2, int(min(h, w) * 0.006))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=1)
        mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)
    border = max(1, int(min(h, w) * 0.005))
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0
    return mask


def _specific_red_mask_rgb(rgb: np.ndarray) -> np.ndarray:
    """Strong detector for faded red/pink digits, where grayscale threshold often fails."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    h = hsv[:, :, 0].astype(np.uint8)
    s = hsv[:, :, 1].astype(np.uint8)
    a = lab[:, :, 1].astype(np.int16)

    red_dom = np.maximum(0, r - np.maximum(g, b)).astype(np.float32)
    red_axis = np.maximum(0, a - 128).astype(np.float32)
    score = red_dom * 4.0 + red_axis * 2.5 + s.astype(np.float32) * 0.35
    score_u8 = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    score_u8 = cv2.GaussianBlur(score_u8, (3, 3), 0)

    _, m_otsu = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hue_red = ((h <= 18) | (h >= 160)) & (s >= 10) & (red_dom >= 4)
    q = float(np.percentile(score_u8, 93))
    m_pct = ((score_u8 >= max(20.0, q)) & (red_dom >= 4)).astype(np.uint8) * 255
    return cv2.bitwise_or(cv2.bitwise_or(m_otsu, hue_red.astype(np.uint8) * 255), m_pct)


def _generic_colored_ink_mask_rgb(rgb: np.ndarray) -> np.ndarray:
    """Detect non-gray colored inks: red, blue, green, purple, orange, pink."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    chroma = (maxc - minc).astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)

    # Lab chroma catches color difference even when HSV saturation is weak.
    a = lab[:, :, 1].astype(np.float32) - 128.0
    bb = lab[:, :, 2].astype(np.float32) - 128.0
    lab_chroma = np.sqrt(a * a + bb * bb)

    score = chroma * 2.2 + sat * 0.9 + lab_chroma * 1.4
    score_u8 = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    score_u8 = cv2.GaussianBlur(score_u8, (3, 3), 0)

    # Percentile threshold keeps only the strongest colored strokes.
    q = float(np.percentile(score_u8, 92))
    mask1 = (score_u8 >= max(26.0, q)).astype(np.uint8) * 255

    # Direct saturation threshold for clean blue/green/red marker.
    mask2 = ((hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 35)).astype(np.uint8) * 255

    return cv2.bitwise_or(mask1, mask2)


def _color_ink_mask(image: np.ndarray) -> np.ndarray:
    """
    Color-ink mask under both RGB and BGR assumptions.
    Your project sometimes sends RGB from PIL and sometimes BGR from OpenCV.
    """
    arr = _to_rgb(image)
    candidates = []
    for rgb in (arr, arr[:, :, ::-1]):
        candidates.append(_specific_red_mask_rgb(rgb))
        candidates.append(_generic_colored_ink_mask_rgb(rgb))

    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    for m in candidates:
        mask = cv2.bitwise_or(mask, m)
    mask = _clean_mask(mask, arr.shape[:2], connect=True)

    ratio = cv2.countNonZero(mask) / float(mask.shape[0] * mask.shape[1])
    if ratio < 0.0003 or ratio > 0.45:
        return np.zeros_like(mask)
    return mask


def _low_contrast_dark_ink_mask(gray: np.ndarray) -> np.ndarray:
    """Detect pencil / light black strokes on bright paper using local background subtraction."""
    h, w = gray.shape[:2]
    # Big blur approximates page/background brightness.
    k = max(31, int(min(h, w) * 0.09) | 1)
    bg = cv2.GaussianBlur(gray, (k, k), 0)
    diff = cv2.subtract(bg, gray)  # positive where strokes are darker than page
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    diff = cv2.GaussianBlur(diff, (3, 3), 0)
    _, m1 = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    q = np.percentile(diff, 92)
    m2 = (diff >= max(15, q)).astype(np.uint8) * 255
    return _clean_mask(cv2.bitwise_or(m1, m2), gray.shape[:2], connect=True)


def _light_ink_on_dark_mask(gray: np.ndarray) -> np.ndarray:
    """Detect white/chalk digits on black/dark background."""
    h, w = gray.shape[:2]
    k = max(31, int(min(h, w) * 0.09) | 1)
    bg = cv2.GaussianBlur(gray, (k, k), 0)
    diff = cv2.subtract(gray, bg)  # positive where digits are lighter than background
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m1 = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    q = np.percentile(diff, 92)
    m2 = (diff >= max(15, q)).astype(np.uint8) * 255
    return _clean_mask(cv2.bitwise_or(m1, m2), gray.shape[:2], connect=True)


def _threshold_masks(gray: np.ndarray) -> List[np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.bilateralFilter(enhanced, 7, 50, 50)

    masks: List[np.ndarray] = []
    _, otsu_inv = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, otsu_norm = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.extend([otsu_inv, otsu_norm])

    for block in (31, 45, 61):
        if block >= min(gray.shape[:2]):
            continue
        if block % 2 == 0:
            block += 1
        masks.append(cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, block, 9))
        masks.append(cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block, 9))

    masks.append(_low_contrast_dark_ink_mask(enhanced))
    masks.append(_light_ink_on_dark_mask(enhanced))
    return [_clean_mask(m, gray.shape[:2], connect=True) for m in masks]


def _is_plausible_box(box: Box, fg_area: int, image_area: int, h_img: int, w_img: int) -> bool:
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return False
    area = w * h
    aspect = w / float(h)
    min_area = max(18, int(image_area * 0.00010))
    max_area = int(image_area * 0.65)
    min_h = max(7, int(h_img * 0.012))
    min_w = max(2, int(w_img * 0.0025))
    if area < min_area or area > max_area:
        return False
    if h < min_h or w < min_w:
        return False
    if fg_area < max(6, int(area * 0.010)):
        return False
    if aspect > 9.5 or aspect < 0.035:
        return False
    if w > 0.97 * w_img and h > 0.97 * h_img:
        return False
    return True


def _initial_boxes(mask: np.ndarray) -> List[Box]:
    h_img, w_img = mask.shape[:2]
    image_area = h_img * w_img
    boxes: List[Box] = []

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if _is_plausible_box((int(x), int(y), int(w), int(h)), int(area), image_area, h_img, w_img):
            boxes.append((int(x), int(y), int(w), int(h)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        fg = int(cv2.countNonZero(mask[y:y+h, x:x+w]))
        if _is_plausible_box((x, y, w, h), fg, image_area, h_img, w_img):
            boxes.append((x, y, w, h))
    return _deduplicate_boxes(boxes, 0.80)


def _iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _deduplicate_boxes(boxes: Sequence[Box], iou_threshold: float = 0.85) -> List[Box]:
    result: List[Box] = []
    for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
        if all(_iou(box, kept) < iou_threshold for kept in result):
            result.append(box)
    return result


def _score_mask(mask: np.ndarray) -> float:
    h, w = mask.shape[:2]
    area = h * w
    ratio = cv2.countNonZero(mask) / float(area)
    if ratio < 0.0004 or ratio > 0.45:
        return -1e9
    boxes = _initial_boxes(mask)
    if not boxes:
        return -1e9
    n = len(boxes)
    heights = np.array([b[3] for b in boxes], dtype=np.float32)
    areas = np.array([b[2] * b[3] for b in boxes], dtype=np.float32)
    median_h = float(np.median(heights))
    height_consistency = 1.0 / (1.0 + float(np.std(heights) / max(median_h, 1.0)))
    # Favor a realistic digit count. A mask with 60-100 tiny components is usually
    # paper noise, not digits. Do not reward more boxes forever.
    if n <= 20:
        n_score = n * 9.0
    else:
        n_score = 20 * 9.0 - (n - 20) * 12.0

    # Digits usually occupy a small/moderate part of the full image.
    ratio_score = 35.0 * np.exp(-abs(ratio - 0.065) * 16.0)
    consistency_score = 30.0 * height_consistency
    blob_penalty = sum(1 for a in areas if a > area * 0.20) * 60.0
    many_box_penalty = max(0, n - 35) * 20.0
    return float(n_score + ratio_score + consistency_score - blob_penalty - many_box_penalty)


def build_foreground_mask(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (mask, enhanced_gray) where digits are white on black.
    This function tries color, dark ink, light ink, Otsu and adaptive masks,
    then automatically chooses the mask that gives digit-like components.
    """
    image = auto_orient_for_multidigit_line(image)
    gray = _to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = cv2.bilateralFilter(clahe.apply(gray), 7, 50, 50)

    candidate_masks = []
    color_mask = _color_ink_mask(image)
    if cv2.countNonZero(color_mask) > 0:
        candidate_masks.append(color_mask)
    candidate_masks.extend(_threshold_masks(gray))

    # Also test union of color mask + low contrast dark mask for faded colored pencil/marker.
    if cv2.countNonZero(color_mask) > 0:
        candidate_masks.append(_clean_mask(cv2.bitwise_or(color_mask, _low_contrast_dark_ink_mask(gray)), gray.shape[:2], True))

    best_mask = None
    best_score = -1e18
    for m in candidate_masks:
        s = _score_mask(m)
        if s > best_score:
            best_score = s
            best_mask = m

    if best_mask is None or best_score < -1e8:
        # Safe fallback for white paper with dark text.
        _, best_mask = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        best_mask = _clean_mask(best_mask, gray.shape[:2], True)

    return best_mask, enhanced_gray


def _merge_overlapping_or_broken_parts(boxes: Sequence[Box]) -> List[Box]:
    boxes = list(boxes)
    if not boxes:
        return []
    median_h = float(np.median([b[3] for b in boxes]))
    gap = max(2, int(median_h * 0.10))

    changed = True
    while changed:
        changed = False
        used = [False] * len(boxes)
        merged: List[Box] = []
        for i, a in enumerate(boxes):
            if used[i]:
                continue
            ax, ay, aw, ah = a
            nx1, ny1, nx2, ny2 = ax, ay, ax + aw, ay + ah
            used[i] = True
            for j, b in enumerate(boxes):
                if used[j] or i == j:
                    continue
                bx, by, bw, bh = b
                overlap_y = min(ny2, by + bh) - max(ny1, by)
                same_line = abs((ay + ah / 2) - (by + bh / 2)) < max(ah, bh) * 0.42
                close_x = bx <= nx2 + gap and nx1 <= bx + bw + gap
                enough_y = overlap_y > min(ah, bh) * 0.22
                # Merge only broken pieces, not normal digit spacing.
                if same_line and close_x and enough_y:
                    nx1, ny1 = min(nx1, bx), min(ny1, by)
                    nx2, ny2 = max(nx2, bx + bw), max(ny2, by + bh)
                    used[j] = True
                    changed = True
            merged.append((int(nx1), int(ny1), int(nx2 - nx1), int(ny2 - ny1)))
        boxes = merged
    return boxes


def _split_box_by_vertical_projection(mask: np.ndarray, box: Box, min_digit_width: int = 5) -> List[Box]:
    x, y, w, h = box
    aspect = w / float(max(h, 1))
    if aspect < 1.14 or w < min_digit_width * 2:
        return [box]
    crop = mask[y:y+h, x:x+w]
    projection = crop.sum(axis=0).astype(np.float32) / 255.0
    if projection.max() <= 0:
        return [box]
    k = max(3, int(w * 0.04))
    if k % 2 == 0:
        k += 1
    projection = cv2.GaussianBlur(projection.reshape(1, -1), (k, 1), 0).ravel()
    threshold = max(1.0, projection.max() * 0.10)
    low_cols = np.where(projection <= threshold)[0]
    low_cols = low_cols[(low_cols > int(w * 0.10)) & (low_cols < int(w * 0.90))]
    if len(low_cols) == 0:
        return [box]
    groups = np.split(low_cols, np.where(np.diff(low_cols) > 1)[0] + 1)
    split_points: List[int] = []
    for g in groups:
        if len(g) >= max(1, int(w * 0.012)):
            sp = int(np.mean(g))
            if sp >= min_digit_width and (w - sp) >= min_digit_width:
                split_points.append(sp)
    if not split_points:
        return [box]

    expected_parts = int(np.clip(round(aspect / 0.60), 2, 8))
    max_splits = expected_parts - 1
    if len(split_points) > max_splits:
        targets = np.linspace(w / expected_parts, w - w / expected_parts, max_splits)
        chosen = []
        available = split_points[:]
        for t in targets:
            best = min(available, key=lambda s: abs(s - t))
            chosen.append(best)
            available.remove(best)
        split_points = sorted(chosen)

    parts: List[Box] = []
    prev = 0
    for sp in sorted(split_points) + [w]:
        if sp - prev >= min_digit_width:
            parts.append((x + prev, y, sp - prev, h))
        prev = sp
    return parts if len(parts) > 1 else [box]


def _split_wide_boxes(mask: np.ndarray, boxes: Sequence[Box]) -> List[Box]:
    final: List[Box] = []
    queue = list(boxes)
    while queue:
        box = queue.pop(0)
        parts = _split_box_by_vertical_projection(mask, box)
        if len(parts) == 1:
            final.append(box)
        else:
            for p in parts:
                if p[2] / float(max(p[3], 1)) > 1.25:
                    queue.append(p)
                else:
                    final.append(p)
    return final


def _filter_tiny_outlier_boxes(boxes: Sequence[Box]) -> List[Box]:
    boxes = list(boxes)
    if len(boxes) <= 2:
        return boxes
    heights = np.array([b[3] for b in boxes], dtype=np.float32)
    areas = np.array([b[2] * b[3] for b in boxes], dtype=np.float32)
    median_h = float(np.median(heights))
    median_area = float(np.median(areas))
    filtered: List[Box] = []
    for b, h, area in zip(boxes, heights, areas):
        # Keep thin 1s, remove page specks.
        if median_h >= 35 and h < median_h * 0.32:
            continue
        if median_area >= 800 and area < median_area * 0.055:
            continue
        filtered.append(b)
    return filtered


def _sort_row_wise(boxes: Sequence[Box]) -> List[Box]:
    boxes = sorted(boxes, key=lambda b: (b[1] + b[3] / 2, b[0]))
    if not boxes:
        return []
    median_h = float(np.median([b[3] for b in boxes]))
    row_threshold = max(12.0, median_h * 0.65)
    rows: List[List[Box]] = []
    for box in boxes:
        cy = box[1] + box[3] / 2.0
        for row in rows:
            row_cy = np.mean([b[1] + b[3] / 2.0 for b in row])
            if abs(cy - row_cy) <= row_threshold:
                row.append(box)
                break
        else:
            rows.append([box])
    rows.sort(key=lambda row: np.mean([b[1] + b[3] / 2.0 for b in row]))
    out: List[Box] = []
    for row in rows:
        out.extend(sorted(row, key=lambda b: b[0]))
    return out


def prepare_mnist_canvas(mask: np.ndarray, box: Box, canvas_size: int = 28, digit_size: int = 20) -> DigitCandidate | None:
    x, y, w, h = box
    pad = int(max(w, h) * 0.18) + 4
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(mask.shape[1], x + w + pad)
    y2 = min(mask.shape[0], y + h + pad)
    crop = mask[y1:y2, x1:x2]
    ys, xs = np.where(crop > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    crop = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ch, cw = crop.shape[:2]
    if ch <= 0 or cw <= 0:
        return None

    scale = digit_size / float(max(ch, cw))
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    x_off = (canvas_size - new_w) // 2
    y_off = (canvas_size - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    moments = cv2.moments(canvas)
    if moments["m00"] > 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        mat = np.float32([[1, 0, canvas_size / 2.0 - cx], [0, 1, canvas_size / 2.0 - cy]])
        canvas = cv2.warpAffine(canvas, mat, (canvas_size, canvas_size), borderValue=0)

    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    return DigitCandidate(box=box, canvas28=canvas, crop=crop)


def detect_and_prepare_digits(image: np.ndarray, max_digits: int = 80) -> Tuple[List[DigitCandidate], np.ndarray, np.ndarray]:
    mask, enhanced_gray = build_foreground_mask(image)
    boxes = _initial_boxes(mask)
    boxes = _merge_overlapping_or_broken_parts(boxes)
    boxes = _split_wide_boxes(mask, boxes)
    boxes = _deduplicate_boxes(boxes, 0.75)
    boxes = _filter_tiny_outlier_boxes(boxes)
    boxes = _sort_row_wise(boxes)

    candidates: List[DigitCandidate] = []
    for box in boxes[:max_digits]:
        cand = prepare_mnist_canvas(mask, box)
        if cand is not None:
            candidates.append(cand)
    return candidates, mask, enhanced_gray


def draw_digit_boxes(image: np.ndarray, predictions: Sequence[dict]) -> np.ndarray:
    arr = np.asarray(image).copy()
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3]

    for item in predictions:
        x, y, w, h = [int(v) for v in item["box"]]
        digit = item.get("digit", "?")
        conf = float(item.get("confidence", 0.0))
        color = (0, 180, 0) if conf >= 80 else (255, 160, 0) if conf >= 50 else (220, 0, 0)
        cv2.rectangle(arr, (x, y), (x + w, y + h), color, 2)
        cv2.putText(arr, f"{digit} {conf:.0f}%", (x, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return arr


# ------------------------------------------------------------
# Worksheet-only digit detector helpers
# These are kept separate so they do not overwrite the main _to_gray() above.
# ------------------------------------------------------------

def _to_gray_worksheet(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def _prepare_binary_for_worksheet(img_bgr):
    gray = _to_gray_worksheet(img_bgr)

    # Improve contrast
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu inverse threshold
    _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Adaptive threshold fallback
    th2 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 12
    )

    # Choose better mask
    if np.count_nonzero(th1) > np.count_nonzero(th2) * 1.8:
        binary = th2
    else:
        binary = th1

    # Remove tiny noise
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


def _prepare_digit_crop(binary, x, y, w, h, out_size=28):
    crop = binary[y:y+h, x:x+w]

    # Add padding
    pad = 6
    crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    # Resize while keeping aspect ratio
    h0, w0 = crop.shape[:2]
    if h0 == 0 or w0 == 0:
        return np.zeros((28, 28), dtype=np.float32)

    scale = 20.0 / max(h0, w0)
    new_w = max(1, int(w0 * scale))
    new_h = max(1, int(h0 * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((out_size, out_size), dtype=np.uint8)
    x_off = (out_size - new_w) // 2
    y_off = (out_size - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    canvas = canvas.astype("float32") / 255.0
    return canvas


def _group_boxes_into_rows(boxes, y_tol=20):
    """
    boxes: list of dicts with keys x,y,w,h,...
    returns row-wise grouped boxes
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: (b["y"], b["x"]))
    rows = []

    for box in boxes:
        cy = box["y"] + box["h"] // 2
        placed = False

        for row in rows:
            row_cy = int(np.mean([b["y"] + b["h"] // 2 for b in row]))
            if abs(cy - row_cy) <= y_tol:
                row.append(box)
                placed = True
                break

        if not placed:
            rows.append([box])

    for row in rows:
        row.sort(key=lambda b: b["x"])

    rows.sort(key=lambda r: min(b["y"] for b in r))
    return rows


def detect_worksheet_digits_only(image_bgr, model, confidence_threshold=0.70):
    """
    Detect only digits from worksheet-like images.
    Ignores x and = based on confidence + shape filtering.
    """
    binary = _prepare_binary_for_worksheet(image_bgr)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    candidates = []
    vis = image_bgr.copy()

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < 40:
            continue
        if w < 5 or h < 10:
            continue
        if h > image_bgr.shape[0] * 0.4:
            continue

        aspect = w / float(h)
        fill_ratio = area / float(w * h + 1e-6)

        # Remove obvious "=" like flat shapes
        if aspect > 3.5 and h < 25:
            continue

        # Remove tiny horizontal noise
        if h < 8:
            continue

        digit_img = _prepare_digit_crop(binary, x, y, w, h)
        candidates.append({
            "x": x, "y": y, "w": w, "h": h,
            "area": area,
            "aspect": aspect,
            "fill_ratio": fill_ratio,
            "img": digit_img
        })

    if not candidates:
        return {
            "rows": [],
            "all_digits": [],
            "binary": binary,
            "visualized": vis
        }

    batch = np.array([c["img"] for c in candidates], dtype=np.float32)
    batch = batch[..., np.newaxis]

    preds = model.predict(batch, verbose=0)

    final_boxes = []
    for c, p in zip(candidates, preds):
        digit = int(np.argmax(p))
        conf = float(np.max(p))

        # Ignore x and = because digit model usually gives lower confidence on them
        if conf < confidence_threshold:
            continue

        c["digit"] = digit
        c["confidence"] = conf
        final_boxes.append(c)

        cv2.rectangle(vis, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f'{digit} ({conf:.2f})',
            (c["x"], max(15, c["y"] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    if final_boxes:
        median_h = int(np.median([b["h"] for b in final_boxes]))
        row_tol = max(15, median_h // 2)
    else:
        row_tol = 20

    rows = _group_boxes_into_rows(final_boxes, y_tol=row_tol)

    return {
        "rows": rows,
        "all_digits": final_boxes,
        "binary": binary,
        "visualized": vis
    }

# ============================================================
# OVERRIDE FIX: cleaner worksheet / multi-digit sequence detector
# Added to fix duplicate boxes and broken digit parts in 1234567890 style images.
# This overrides _prepare_binary_for_worksheet() and detect_worksheet_digits_only().
# ============================================================
from typing import Any, Dict


def _box_iou_safe(a, b):
    ax, ay, aw, ah = [int(v) for v in a]
    bx, by, bw, bh = [int(v) for v in b]
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(1, aw * ah)
    area_b = max(1, bw * bh)
    return inter / float(area_a + area_b - inter + 1e-6), inter / float(min(area_a, area_b) + 1e-6)


def _merge_two_boxes(a, b):
    ax, ay, aw, ah = [int(v) for v in a]
    bx, by, bw, bh = [int(v) for v in b]
    x1, y1 = min(ax, bx), min(ay, by)
    x2, y2 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
    return (x1, y1, x2 - x1, y2 - y1)


def _remove_border_components_for_worksheet(mask):
    """Remove photo/UI border shadows while keeping real digits near the edges."""
    H, W = mask.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    clean = np.zeros_like(mask, dtype=np.uint8)

    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        touches = int(x <= 2) + int(y <= 2) + int(x + w >= W - 3) + int(y + h >= H - 3)

        # Top/bottom/side dark strips often come from camera frames or Streamlit screenshots.
        long_horizontal_strip = (y <= 2 or y + h >= H - 3) and w > W * 0.10 and h < max(25, H * 0.12)
        long_vertical_strip = (x <= 2 or x + w >= W - 3) and h > H * 0.10 and w < max(25, W * 0.025)
        huge_component = w > W * 0.70 or h > H * 0.70

        if huge_component or long_horizontal_strip or long_vertical_strip:
            continue
        if touches >= 2 and area > max(50, int(H * W * 0.0003)):
            continue

        clean[labels == i] = 255

    return clean


def _prepare_binary_for_worksheet(img_bgr):
    """
    Cleaner mask for phone/scanned worksheet images.

    The old version used Otsu + adaptive threshold directly on the page. On
    shadowed paper that can turn background texture into foreground, so the app
    detects many fake digits. This version uses local background subtraction:
    only strokes that are darker/lighter than their nearby paper/background are
    kept.
    """
    gray = _to_gray_worksheet(img_bgr)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    H, W = gray.shape[:2]

    # Estimate local paper/background brightness. A big blur follows lighting
    # changes but not pen strokes.
    k = max(31, int(min(H, W) * 0.12) | 1)
    bg = cv2.GaussianBlur(gray, (k, k), 0)

    dark_diff = cv2.subtract(bg, gray)   # dark ink on light/normal paper
    light_diff = cv2.subtract(gray, bg)  # white/chalk digits on dark background

    def _mask_from_local_diff(diff, min_thr=12.0):
        _, otsu = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pct = float(np.percentile(diff, 97.5))
        pct_mask = (diff >= max(min_thr, pct)).astype(np.uint8) * 255
        candidate = cv2.bitwise_or(otsu, pct_mask)

        # If Otsu becomes too greedy, keep only strongest local-contrast pixels.
        ratio = cv2.countNonZero(candidate) / float(max(1, H * W))
        if ratio > 0.16:
            candidate = pct_mask
        return candidate

    dark_mask = _mask_from_local_diff(dark_diff, 12.0)
    light_mask = _mask_from_local_diff(light_diff, 12.0)

    # Colored handwriting: require saturation AND local darkness. This prevents
    # brown/yellow paper from being detected as colored digits.
    if np.asarray(img_bgr).ndim == 3:
        if img_bgr.shape[2] == 4:
            rgb = cv2.cvtColor(np.asarray(img_bgr).astype(np.uint8), cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(np.asarray(img_bgr).astype(np.uint8), cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        color_mask = ((s > 35) & (dark_diff > 8) & (v > 25)).astype(np.uint8) * 255
    else:
        color_mask = np.zeros_like(gray, dtype=np.uint8)

    dark_background = (float(np.mean(gray < 80)) > 0.20) or (float(np.mean(gray)) < 115)
    binary = cv2.bitwise_or(light_mask if dark_background else dark_mask, color_mask)

    # Clean without joining separate digits. One small dilation keeps thin 1/7
    # strokes visible, but avoids merging spaced digits.
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    binary = cv2.dilate(binary, np.ones((2, 2), np.uint8), iterations=1)
    binary = _remove_border_components_for_worksheet(binary)

    return binary


def _candidate_boxes_from_binary(binary):
    H, W = binary.shape[:2]
    img_area = H * W
    boxes = []

    # External contours reduce duplicate/nested boxes.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area <= 0:
            area = cv2.countNonZero(binary[y:y+h, x:x+w])
        aspect = w / float(max(1, h))
        fill = area / float(max(1, w * h))

        # Dynamic filters. Keep thin 1, reject noise and huge full-image boxes.
        if area < max(25, img_area * 0.00008):
            continue
        if h < max(12, H * 0.035):
            continue
        if w < max(3, W * 0.004):
            continue
        if h > H * 0.85 or w > W * 0.40:
            continue
        if aspect > 2.4 and h < H * 0.30:
            continue
        if aspect < 0.035:
            continue
        if fill < 0.018:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))

    return boxes


def _merge_broken_digit_boxes_plain(boxes, image_shape):
    """Merge pieces belonging to one broken digit, but avoid merging normal neighboring digits."""
    boxes = [tuple(map(int, b)) for b in boxes]
    if not boxes:
        return []

    H, W = image_shape[:2]
    changed = True
    while changed:
        changed = False
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        used = [False] * len(boxes)
        merged = []

        for i, a in enumerate(boxes):
            if used[i]:
                continue
            cur = a
            used[i] = True
            cx, cy, cw, ch = cur
            for j, b in enumerate(boxes):
                if used[j]:
                    continue
                bx, by, bw, bh = b

                iou, iom = _box_iou_safe(cur, b)

                # Horizontal and vertical relationship.
                cur_x, cur_y, cur_w, cur_h = cur
                x_overlap = min(cur_x + cur_w, bx + bw) - max(cur_x, bx)
                y_overlap = min(cur_y + cur_h, by + bh) - max(cur_y, by)
                x_overlap_ratio = x_overlap / float(max(1, min(cur_w, bw)))
                y_overlap_ratio = y_overlap / float(max(1, min(cur_h, bh)))
                v_gap = max(0, max(cur_y, by) - min(cur_y + cur_h, by + bh))
                h_gap = max(0, max(cur_x, bx) - min(cur_x + cur_w, bx + bw))

                same_digit_vertical_piece = (
                    x_overlap_ratio > 0.18 and
                    v_gap <= max(5, int(max(cur_h, bh) * 0.28)) and
                    h_gap <= max(6, int(max(cur_w, bw) * 0.35))
                )

                duplicate_or_nested = iou > 0.18 or iom > 0.60

                # Do not merge two normal neighboring digits: they have little x-overlap and similar height.
                if duplicate_or_nested or same_digit_vertical_piece:
                    cur = _merge_two_boxes(cur, b)
                    used[j] = True
                    changed = True
            merged.append(cur)
        boxes = merged

    return boxes


def _remove_duplicate_plain_boxes(boxes):
    """Keep one box when duplicate/nested boxes remain after merge."""
    keep = []
    boxes = sorted([tuple(map(int, b)) for b in boxes], key=lambda b: b[2] * b[3], reverse=True)
    for b in boxes:
        duplicate = False
        for k in keep:
            iou, iom = _box_iou_safe(b, k)
            if iou > 0.25 or iom > 0.72:
                duplicate = True
                break
        if not duplicate:
            keep.append(b)
    return keep


def _filter_plain_boxes_by_row_stats(boxes):
    if len(boxes) <= 2:
        return boxes
    heights = np.array([b[3] for b in boxes], dtype=np.float32)
    widths = np.array([b[2] for b in boxes], dtype=np.float32)
    areas = np.array([b[2] * b[3] for b in boxes], dtype=np.float32)
    med_h = float(np.median(heights))
    med_area = float(np.median(areas))
    out = []
    for b, h, w, area in zip(boxes, heights, widths, areas):
        aspect = w / float(max(1, h))
        # Keep thin 1; remove small fragments.
        if h < med_h * 0.42 and aspect > 0.18:
            continue
        if area < med_area * 0.12 and aspect > 0.18:
            continue
        out.append(b)
    return out


def _filter_plain_boxes_by_primary_rows(boxes):
    """
    Remove off-row paper texture/noise. Keeps rows whose digit height is close
    to the strongest row, so multi-row worksheets still work.
    """
    if len(boxes) <= 2:
        return boxes

    med_h_all = float(np.median([b[3] for b in boxes]))
    row_gap = max(12.0, med_h_all * 0.70)
    rows = []
    for b in sorted(boxes, key=lambda x: x[1] + x[3] / 2.0):
        cy = b[1] + b[3] / 2.0
        placed = False
        for row in rows:
            row_cy = np.mean([r[1] + r[3] / 2.0 for r in row])
            if abs(cy - row_cy) <= row_gap:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])

    if not rows:
        return boxes

    row_median_h = [float(np.median([b[3] for b in row])) for row in rows]
    row_median_area = [float(np.median([b[2] * b[3] for b in row])) for row in rows]
    best_h = max(row_median_h)
    best_area = max(row_median_area)

    kept = []
    for row, mh, ma in zip(rows, row_median_h, row_median_area):
        if mh >= best_h * 0.52 and ma >= best_area * 0.18:
            kept.extend(row)

    return kept if kept else boxes



def _sort_plain_boxes_rowwise(boxes):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[1] + b[3] / 2, b[0]))
    med_h = float(np.median([b[3] for b in boxes]))
    row_gap = max(12.0, med_h * 0.62)
    rows = []
    for b in boxes:
        cy = b[1] + b[3] / 2.0
        placed = False
        for row in rows:
            row_cy = np.mean([r[1] + r[3] / 2.0 for r in row])
            if abs(cy - row_cy) <= row_gap:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])
    rows.sort(key=lambda r: np.mean([b[1] + b[3] / 2.0 for b in r]))
    out = []
    for row in rows:
        out.extend(sorted(row, key=lambda b: b[0]))
    return out


def _prepare_digit_crop_uint8(binary, box, out_size=28, digit_size=20):
    x, y, w, h = [int(v) for v in box]
    H, W = binary.shape[:2]
    pad = int(max(w, h) * 0.18) + 4
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    crop = binary[y1:y2, x1:x2]
    pts = cv2.findNonZero(crop)
    if pts is None:
        return None
    rx, ry, rw, rh = cv2.boundingRect(pts)
    crop = crop[ry:ry + rh, rx:rx + rw]
    ch, cw = crop.shape[:2]
    if ch <= 0 or cw <= 0:
        return None
    scale = digit_size / float(max(ch, cw))
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_size, out_size), dtype=np.uint8)
    xo = (out_size - new_w) // 2
    yo = (out_size - new_h) // 2
    canvas[yo:yo + new_h, xo:xo + new_w] = resized

    # MNIST-style centering by center of mass.
    m = cv2.moments(canvas)
    if m["m00"] > 0:
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        mat = np.float32([[1, 0, out_size / 2.0 - cx], [0, 1, out_size / 2.0 - cy]])
        canvas = cv2.warpAffine(canvas, mat, (out_size, out_size), borderValue=0)
    return canvas


def _model_input_hw_c(model):
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list):
        shape = shape[0]
    try:
        return int(shape[1]), int(shape[2]), int(shape[3])
    except Exception:
        return 28, 28, 1


def _canvas_to_model_input(canvas28, model):
    h, w, c = _model_input_hw_c(model)
    img = cv2.resize(canvas28, (w, h), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
    if c == 1:
        return img.reshape(h, w, 1)
    img3 = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB).astype("float32") / 255.0
    return img3.reshape(h, w, 3)


def detect_worksheet_digits_only(image_bgr, model, confidence_threshold=0.70):
    """
    Improved worksheet / multi-digit detector.
    Fixes:
      - duplicate nested boxes
      - broken 6/8/9/0 parts being predicted as separate digits
      - wrong input shape for CNN models expecting 32x32x3
    Returns the same keys expected by backend/services/prediction_service.py.
    """
    if image_bgr is None:
        return {"rows": [], "all_digits": [], "binary": None, "visualized": None}

    img = np.asarray(image_bgr).astype(np.uint8)
    binary = _prepare_binary_for_worksheet(img)

    boxes = _candidate_boxes_from_binary(binary)
    boxes = _merge_broken_digit_boxes_plain(boxes, binary.shape)
    boxes = _remove_duplicate_plain_boxes(boxes)
    boxes = _filter_plain_boxes_by_row_stats(boxes)
    boxes = _filter_plain_boxes_by_primary_rows(boxes)
    boxes = _sort_plain_boxes_rowwise(boxes)

    candidates = []
    for box in boxes:
        canvas28 = _prepare_digit_crop_uint8(binary, box)
        if canvas28 is None:
            continue
        x, y, w, h = box
        candidates.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "box": (int(x), int(y), int(w), int(h)),
            "canvas28": canvas28,
        })

    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if not candidates:
        return {"rows": [], "all_digits": [], "binary": binary, "visualized": vis}

    batch = np.array([_canvas_to_model_input(c["canvas28"], model) for c in candidates], dtype=np.float32)
    preds = model.predict(batch, verbose=0)

    final_boxes = []
    for c, p in zip(candidates, preds):
        digit = int(np.argmax(p))
        conf = float(np.max(p))
        # Keep the detection even when confidence is low. The caller marks it
        # as low_confidence; falling back to the general detector often creates
        # many false digit boxes on camera photos.
        top3_idx = np.argsort(p)[-3:][::-1]
        top3 = [{"digit": int(i), "confidence": round(float(p[i]) * 100, 2)} for i in top3_idx]
        c["digit"] = digit
        c["confidence"] = conf
        c["top3_predictions"] = top3
        c["status"] = "ok" if conf >= float(confidence_threshold) else "low_confidence"
        final_boxes.append(c)

    # Final duplicate removal after prediction: if two boxes still overlap, keep higher confidence/larger box.
    final_boxes = sorted(final_boxes, key=lambda c: (float(c.get("confidence", 0.0)), c["w"] * c["h"]), reverse=True)
    kept = []
    for c in final_boxes:
        b = c["box"]
        if all(_box_iou_safe(b, k["box"])[1] < 0.65 and _box_iou_safe(b, k["box"])[0] < 0.25 for k in kept):
            kept.append(c)
    final_boxes = _group_boxes_into_rows(kept, y_tol=max(15, int(np.median([c["h"] for c in kept]) * 0.65)) if kept else 20)
    flat = [c for row in final_boxes for c in row]

    # Draw in final row-wise order.
    for pos, c in enumerate(flat, start=1):
        x, y, w, h = c["box"]
        digit = c["digit"]
        conf = c["confidence"] * 100.0
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(vis, f"{digit} {conf:.0f}%", (x, max(16, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2, cv2.LINE_AA)

    return {
        "rows": final_boxes,
        "all_digits": flat,
        "binary": binary,
        "visualized": vis,
    }
