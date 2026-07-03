"""
Universal multi-digit detector for DigitAI.

Goal:
- Detect individual digit-like symbols from many image styles:
  black/blue/green/red/yellow handwriting, pencil/faded writing,
  white digits on dark background, white digits inside colored shapes,
  multiple rows, and transparent/checkerboard screenshots.

Note:
This file handles DETECTION/SEGMENTATION. Final class accuracy still depends
on your trained CNN model. If the model was trained only on MNIST, decorative,
font, letter/symbol, or very stylized digits may need retraining.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


Box = Tuple[int, int, int, int]


@dataclass
class DigitCandidate:
    box: Box
    canvas28: np.ndarray
    mask_name: str


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Image is None")
    img = np.asarray(image)
    if img.ndim == 2:
        return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if img.ndim == 3 and img.shape[2] == 4:
        # Composite alpha image on white background.
        rgba = img.astype(np.uint8)
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        rgb = rgba[:, :, :3].astype(np.float32)
        white = np.full_like(rgb, 255.0)
        return (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 3:
        return img.astype(np.uint8)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _remove_checker_noise(gray: np.ndarray) -> np.ndarray:
    """Softly removes very light transparent checkerboard-like backgrounds."""
    # Keep strong foreground and flatten near-white background.
    out = gray.copy()
    out[out > 235] = 255
    return out




def _local_contrast_ink_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    High-precision mask for normal camera photos of handwriting.
    It keeps pixels that are locally darker/lighter than the nearby paper,
    which stops brown paper texture and shadows from becoming fake digits.
    """
    img_rgb = _ensure_rgb(img_rgb)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    H, W = gray.shape[:2]

    k = max(31, int(min(H, W) * 0.12) | 1)
    bg = cv2.GaussianBlur(gray, (k, k), 0)
    dark_diff = cv2.subtract(bg, gray)
    light_diff = cv2.subtract(gray, bg)

    def _mask_from_diff(diff, min_thr=12.0):
        _, otsu = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pct = float(np.percentile(diff, 97.5))
        pct_mask = (diff >= max(min_thr, pct)).astype(np.uint8) * 255
        m = cv2.bitwise_or(otsu, pct_mask)
        if cv2.countNonZero(m) / float(max(1, H * W)) > 0.16:
            m = pct_mask
        return m

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    color_stroke = ((s > 35) & (dark_diff > 8) & (v > 25)).astype(np.uint8) * 255

    dark_background = (float(np.mean(gray < 80)) > 0.20) or (float(np.mean(gray)) < 115)
    base = _mask_from_diff(light_diff if dark_background else dark_diff, 12.0)
    mask = cv2.bitwise_or(base, color_stroke)
    mask = _clean_mask(mask, open_iter=1, close_iter=1, dilate_iter=1)

    # Remove UI/photo border components without deleting real digits near edges.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        touches = int(x <= 2) + int(y <= 2) + int(x + w >= W - 3) + int(y + h >= H - 3)
        long_horizontal = (y <= 2 or y + h >= H - 3) and w > W * 0.10 and h < max(25, H * 0.12)
        long_vertical = (x <= 2 or x + w >= W - 3) and h > H * 0.10 and w < max(25, W * 0.025)
        huge = w > W * 0.70 or h > H * 0.70
        if huge or long_horizontal or long_vertical:
            continue
        if touches >= 2 and area > max(50, int(H * W * 0.0003)):
            continue
        clean[labels == i] = 255

    return clean


def _clean_mask(mask: np.ndarray, open_iter: int = 1, close_iter: int = 1, dilate_iter: int = 0) -> np.ndarray:
    mask = mask.astype(np.uint8)
    k2 = np.ones((2, 2), np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    if open_iter:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k2, iterations=open_iter)
    if close_iter:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=close_iter)
    if dilate_iter:
        mask = cv2.dilate(mask, k2, iterations=dilate_iter)
    return mask


def build_universal_masks(img_rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Builds multiple masks and later the detector chooses useful boxes from all of them.
    Different image types need different masks:
    - colored pen on white background
    - dark pen/pencil on light background
    - white digits on black/colored background
    - faint/blurred scanned digits
    """
    img_rgb = _ensure_rgb(img_rgb)
    blur = cv2.GaussianBlur(img_rgb, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    gray = _remove_checker_noise(gray)

    h, s, v = cv2.split(hsv)

    masks: Dict[str, np.ndarray] = {}

    # High-precision camera-photo mask. This is preferred when it finds a clean row.
    local_ink = _local_contrast_ink_mask(img_rgb)
    if cv2.countNonZero(local_ink) > 0:
        masks["local_dark_ink"] = local_ink

    # 1A) Strict dark strokes: best for clear black/gray thick digits and prevents row merging.
    strict_dark = ((gray < 150) & (s < 100)).astype(np.uint8) * 255
    masks["strict_dark"] = _clean_mask(strict_dark, open_iter=1, close_iter=0, dilate_iter=0)

    # 1B) Dark strokes on bright paper: black pen / pencil / gray stylized digits.
    # Use Otsu + adaptive so both clear and faint strokes work.
    _, dark_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive_dark = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )
    dark = cv2.bitwise_or(dark_otsu, adaptive_dark)
    masks["dark_pen_pencil"] = _clean_mask(dark, open_iter=1, close_iter=1, dilate_iter=0)

    # 2) Saturated colored strokes: red, blue, green, yellow, etc.
    # Remove very low value pixels to avoid shadows.
    color = (((s > 35) & (v > 45))).astype(np.uint8) * 255
    masks["colored_strokes"] = _clean_mask(color, open_iter=1, close_iter=0, dilate_iter=0)

    # 3) White digits on black / dark backgrounds.
    # Enable globally when image has dark background or significant dark area.
    dark_background_ratio = float(np.mean(gray < 80))
    if dark_background_ratio > 0.20 or float(np.mean(gray)) < 135:
        white_on_dark = (((gray > 145) & (s < 140))).astype(np.uint8) * 255
        masks["white_on_dark"] = _clean_mask(white_on_dark, open_iter=1, close_iter=1, dilate_iter=1)

    # 4) White digits inside colored badges/buttons.
    # Use this ONLY when the image contains large filled colored objects.
    # Otherwise, for normal colored handwriting on white paper this mask wrongly grabs the white paper.
    raw_colored_region = (((s > 45) & (v > 45))).astype(np.uint8) * 255
    has_colored_badge = False
    img_area = gray.shape[0] * gray.shape[1]
    for (bx, by, bw, bh), b_area, b_fill, _ in _component_stats(_clean_mask(raw_colored_region, open_iter=1, close_iter=1, dilate_iter=0)):
        aspect_badge = bw / float(bh + 1e-6)
        if b_area > img_area * 0.008 and b_fill > 0.35 and 0.55 <= aspect_badge <= 1.75:
            has_colored_badge = True
            break
    if has_colored_badge:
        # Special badge mode: find white digit strokes inside each filled colored badge,
        # ignoring the outer white circle/ring and watermark pieces near the border.
        badge_digit_mask = np.zeros_like(gray, dtype=np.uint8)
        cleaned_colored = _clean_mask(raw_colored_region, open_iter=1, close_iter=1, dilate_iter=0)
        for (bx, by, bw, bh), b_area, b_fill, _ in _component_stats(cleaned_colored):
            aspect_badge = bw / float(bh + 1e-6)
            if not (b_area > img_area * 0.008 and b_fill > 0.35 and 0.55 <= aspect_badge <= 1.75):
                continue
            mx = int(bw * 0.16)
            my = int(bh * 0.16)
            ix1, iy1 = bx + mx, by + my
            ix2, iy2 = bx + bw - mx, by + bh - my
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            roi_gray = gray[iy1:iy2, ix1:ix2]
            roi_s = s[iy1:iy2, ix1:ix2]
            roi_white = (((roi_gray > 150) & (roi_s < 130))).astype(np.uint8) * 255
            roi_white = _clean_mask(roi_white, open_iter=1, close_iter=1, dilate_iter=1)

            # Keep only large central white components inside the badge.
            # This removes decorative rings/watermark pieces.
            n_roi, lbl_roi, st_roi, cent_roi = cv2.connectedComponentsWithStats(roi_white, connectivity=8)
            comp_ids = []
            max_area_roi = 0
            for ci in range(1, n_roi):
                cx, cy, cw, ch, ca = st_roi[ci]
                ccx, ccy = cent_roi[ci]
                central = (cw >= 5 and ch >= 10 and 0.08 * roi_white.shape[1] <= ccx <= 0.92 * roi_white.shape[1]
                           and 0.05 * roi_white.shape[0] <= ccy <= 0.95 * roi_white.shape[0])
                if central:
                    max_area_roi = max(max_area_roi, int(ca))
            for ci in range(1, n_roi):
                cx, cy, cw, ch, ca = st_roi[ci]
                ccx, ccy = cent_roi[ci]
                if ca < max(18, max_area_roi * 0.22):
                    continue
                if cw < 5 or ch < 10:
                    continue
                if not (0.08 * roi_white.shape[1] <= ccx <= 0.92 * roi_white.shape[1] and 0.05 * roi_white.shape[0] <= ccy <= 0.95 * roi_white.shape[0]):
                    continue
                badge_digit_mask[iy1:iy2, ix1:ix2][lbl_roi == ci] = 255
        masks["badge_white_digit"] = _clean_mask(badge_digit_mask, open_iter=1, close_iter=1, dilate_iter=1)

        # Avoid generic white-inside-color mask in badge images because it also finds rings/watermarks.
        # The dedicated badge_white_digit mask above is cleaner.
        pass

    # 5) Edge-based fallback for decorative / blurred / anti-aliased strokes.
    # Skip it for filled badge images because edges mostly capture circular borders/watermarks.
    if not has_colored_badge:
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        masks["edge_fallback"] = _clean_mask(edges, open_iter=0, close_iter=1, dilate_iter=0)

    return masks



def _split_wide_box_by_projection(mask: np.ndarray, box: Box) -> List[Box]:
    """Split a very wide connected component/row into individual digit boxes using vertical projection gaps."""
    x, y, w, h = box
    if w < h * 1.35 or w < 20:
        return [box]
    crop = mask[y:y+h, x:x+w]
    if crop.size == 0:
        return [box]
    # Projection of foreground pixels per column.
    proj = (crop > 0).sum(axis=0).astype(np.float32)
    if proj.max() <= 0:
        return [box]
    # Smooth projection to avoid tiny anti-aliased gaps.
    k = max(3, min(9, int(round(w * 0.025)) | 1))
    proj_s = cv2.GaussianBlur(proj.reshape(1, -1), (k, 1), 0).ravel()
    low_thr = max(1.0, float(proj_s.max()) * 0.08)
    is_gap = proj_s <= low_thr

    # Make edges gaps to close segments.
    segments: List[Tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for i, gap in enumerate(is_gap):
        if not gap and not in_seg:
            in_seg = True
            seg_start = i
        elif gap and in_seg:
            if i - seg_start >= 3:
                segments.append((seg_start, i))
            in_seg = False
    if in_seg and w - seg_start >= 3:
        segments.append((seg_start, w))

    # If no useful gaps, do not split.
    if len(segments) <= 1:
        return [box]

    out: List[Box] = []
    for a, b in segments:
        sub = crop[:, a:b]
        pts = cv2.findNonZero(sub.astype(np.uint8))
        if pts is None:
            continue
        rx, ry, rw, rh = cv2.boundingRect(pts)
        if rw < 3 or rh < 6:
            continue
        out.append((int(x + a + rx), int(y + ry), int(rw), int(rh)))

    # Avoid bad over-splitting into many tiny pieces.
    if len(out) <= 1:
        return [box]
    return out


def _component_stats(mask: np.ndarray) -> List[Tuple[Box, int, float, float]]:
    """Return (box, area, fill_ratio, border_ratio)."""
    H, W = mask.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w <= 0 or h <= 0:
            continue
        roi = (labels[y:y+h, x:x+w] == i).astype(np.uint8)
        fill = float(area) / float(w * h + 1e-6)
        if w > 4 and h > 4:
            border_pixels = int(roi[0, :].sum() + roi[-1, :].sum() + roi[:, 0].sum() + roi[:, -1].sum())
            border_ratio = float(border_pixels) / float(area + 1e-6)
        else:
            border_ratio = 1.0
        out.append(((int(x), int(y), int(w), int(h)), int(area), fill, border_ratio))
    return out


def _is_reasonable_digit_box(box: Box, area: int, fill: float, border_ratio: float, image_shape: Tuple[int, int], mask_name: str) -> bool:
    H, W = image_shape
    x, y, w, h = box
    img_area = H * W

    min_area = max(10, int(img_area * 0.000015))
    max_area = int(img_area * 0.18)
    if area < min_area or area > max_area:
        return False

    # Very small shapes/noise.
    if w < max(3, int(W * 0.004)) or h < max(7, int(H * 0.012)):
        return False

    aspect = w / float(h + 1e-6)
    if aspect > 2.5 and h < H * 0.18:  # long line/underscore/watermark strip
        return False
    if aspect < 0.05:
        return False

    # Remove large filled circular/solid decorations such as blue badges.
    # Keep real digits, which are usually sparse strokes.
    if mask_name == "colored_strokes" and fill > 0.42 and w > W * 0.05 and h > H * 0.10:
        return False

    # Remove border-only circles/rings from badge images, but keep smaller real zeros.
    if mask_name in {"white_inside_color", "edge_fallback"}:
        near_square = 0.65 <= aspect <= 1.45
        if near_square and border_ratio > 0.35 and w > W * 0.06 and h > H * 0.12:
            return False

    # Remove image-frame/large page border.
    touches_edges = (x <= 1) + (y <= 1) + (x + w >= W - 2) + (y + h >= H - 2)
    if touches_edges >= 2 and (w * h) > img_area * 0.04:
        return False

    return True


def _iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter) / float(union + 1e-6)


def _contains_big(a: Box, b: Box) -> bool:
    """True if a mostly contains b."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter > 0.80 * min(aw * ah, bw * bh)


def _nms_boxes(boxes: List[Tuple[Box, str, int]]) -> List[Tuple[Box, str, int]]:
    if not boxes:
        return []
    # Prefer smaller sparse digit boxes over bigger decorative boxes.
    priority = {
        "badge_white_digit": 0,
        "strict_dark": 1,
        "white_inside_color": 2,
        "white_on_dark": 2,
        "dark_pen_pencil": 3,
        "colored_strokes": 4,
        "edge_fallback": 5,
    }
    boxes = sorted(boxes, key=lambda item: (priority.get(item[1], 9), item[0][2] * item[0][3]))
    keep: List[Tuple[Box, str, int]] = []
    for box, name, area in boxes:
        duplicate = False
        for kept_box, kept_name, kept_area in keep:
            if _iou(box, kept_box) > 0.45 or _contains_big(box, kept_box) or _contains_big(kept_box, box):
                duplicate = True
                break
        if not duplicate:
            keep.append((box, name, area))
    return keep


def _merge_broken_digit_parts(boxes: List[Tuple[Box, str, int]], image_shape: Tuple[int, int]) -> List[Tuple[Box, str, int]]:
    """Conservative merge for broken pencil/script strokes without merging adjacent digits."""
    if not boxes:
        return []
    H, W = image_shape
    items = list(boxes)
    changed = True
    while changed:
        changed = False
        used = [False] * len(items)
        merged: List[Tuple[Box, str, int]] = []
        for i, (b1, n1, a1) in enumerate(items):
            if used[i]:
                continue
            x1, y1, w1, h1 = b1
            bx1, by1, bx2, by2 = x1, y1, x1 + w1, y1 + h1
            area = a1
            name = n1
            for j, (b2, n2, a2) in enumerate(items):
                if i == j or used[j]:
                    continue
                x2, y2, w2, h2 = b2
                cx1, cy1, cx2, cy2 = x2, y2, x2 + w2, y2 + h2
                gap_x = max(0, max(bx1, cx1) - min(bx2, cx2))
                gap_y = max(0, max(by1, cy1) - min(by2, cy2))
                cy_a = (by1 + by2) / 2
                cy_b = (cy1 + cy2) / 2
                same_row = abs(cy_a - cy_b) < max(h1, h2) * 0.45
                vertical_overlap = min(by2, cy2) - max(by1, cy1)
                # Merge only if pieces are very close/overlap vertically; avoid joining separate digits.
                if same_row and vertical_overlap > min(h1, h2) * 0.45 and gap_x <= 2 and gap_y <= 2:
                    bx1, by1 = min(bx1, cx1), min(by1, cy1)
                    bx2, by2 = max(bx2, cx2), max(by2, cy2)
                    area += a2
                    used[j] = True
                    changed = True
            used[i] = True
            merged.append(((int(bx1), int(by1), int(bx2 - bx1), int(by2 - by1)), name, area))
        items = merged
    return items


def sort_boxes_rowwise(boxes: List[Tuple[Box, str, int]]) -> List[Tuple[Box, str, int]]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda item: item[0][1])
    heights = [b[3] for b, _, _ in boxes]
    median_h = float(np.median(heights)) if heights else 10.0
    row_gap = max(8.0, median_h * 0.60)
    rows: List[List[Tuple[Box, str, int]]] = []
    for item in boxes:
        x, y, w, h = item[0]
        cy = y + h / 2
        placed = False
        for row in rows:
            row_cy = np.mean([b[1] + b[3] / 2 for b, _, _ in row])
            if abs(cy - row_cy) <= row_gap:
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])
    out: List[Tuple[Box, str, int]] = []
    for row in rows:
        row.sort(key=lambda item: item[0][0])
        out.extend(row)
    return out



def _filter_universal_items_by_primary_rows(items: List[Tuple[Box, str, int]]) -> List[Tuple[Box, str, int]]:
    if len(items) <= 2:
        return items
    med_h_all = float(np.median([b[3] for b, _, _ in items]))
    row_gap = max(12.0, med_h_all * 0.70)
    rows: List[List[Tuple[Box, str, int]]] = []
    for item in sorted(items, key=lambda it: it[0][1] + it[0][3] / 2.0):
        b = item[0]
        cy = b[1] + b[3] / 2.0
        placed = False
        for row in rows:
            row_cy = np.mean([r[0][1] + r[0][3] / 2.0 for r in row])
            if abs(cy - row_cy) <= row_gap:
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])
    if not rows:
        return items
    row_h = [float(np.median([b[3] for b, _, _ in row])) for row in rows]
    row_area = [float(np.median([b[2] * b[3] for b, _, _ in row])) for row in rows]
    best_h = max(row_h)
    best_area = max(row_area)
    kept: List[Tuple[Box, str, int]] = []
    for row, mh, ma in zip(rows, row_h, row_area):
        if mh >= best_h * 0.52 and ma >= best_area * 0.18:
            kept.extend(row)
    return kept if kept else items


def find_universal_digit_boxes(img_rgb: np.ndarray, max_digits: int = 150) -> Tuple[List[Tuple[Box, str, int]], Dict[str, np.ndarray]]:
    img_rgb = _ensure_rgb(img_rgb)
    H, W = img_rgb.shape[:2]
    masks = build_universal_masks(img_rgb)

    # Prefer the high-precision local-contrast mask when it produces a clean
    # digit row. This prevents broad fallback masks from adding dozens of
    # paper-texture/noise boxes.
    if "local_dark_ink" in masks:
        local_boxes: List[Tuple[Box, str, int]] = []
        local_mask = masks["local_dark_ink"]
        for box, area, fill, border_ratio in _component_stats(local_mask):
            if not _is_reasonable_digit_box(box, area, fill, border_ratio, (H, W), "local_dark_ink"):
                continue
            for sub_box in _split_wide_box_by_projection(local_mask, box):
                sx, sy, sw, sh = sub_box
                if sw < 3 or sh < 6:
                    continue
                local_boxes.append((sub_box, "local_dark_ink", max(1, int(area * (sw * sh) / max(1, box[2] * box[3])))))
        local_boxes = _nms_boxes(local_boxes)
        local_boxes = _merge_broken_digit_parts(local_boxes, (H, W))
        local_boxes = _filter_universal_items_by_primary_rows(local_boxes)
        local_boxes = sort_boxes_rowwise(local_boxes)
        if 2 <= len(local_boxes) <= max_digits:
            return local_boxes[:max_digits], {"local_dark_ink": local_mask}

    all_boxes: List[Tuple[Box, str, int]] = []
    for name, mask in masks.items():
        for box, area, fill, border_ratio in _component_stats(mask):
            if not _is_reasonable_digit_box(box, area, fill, border_ratio, (H, W), name):
                continue
            # Some script rows or close handwriting may appear as one wide component.
            # Split it into digit-like parts using vertical projection gaps.
            for sub_box in _split_wide_box_by_projection(mask, box):
                sx, sy, sw, sh = sub_box
                if sw < 3 or sh < 6:
                    continue
                all_boxes.append((sub_box, name, max(1, int(area * (sw * sh) / max(1, box[2] * box[3])))))
    all_boxes = _nms_boxes(all_boxes)
    all_boxes = _merge_broken_digit_parts(all_boxes, (H, W))
    all_boxes = sort_boxes_rowwise(all_boxes)
    return all_boxes[:max_digits], masks


def _mask_for_box(masks: Dict[str, np.ndarray], mask_name: str) -> np.ndarray:
    # Use selected mask first, but union with close masks when helpful for broken strokes.
    base = masks.get(mask_name)
    if base is None:
        base = next(iter(masks.values()))
    if mask_name in {"strict_dark", "dark_pen_pencil", "colored_strokes"}:
        union = base.copy()
        for k in ("strict_dark", "dark_pen_pencil", "colored_strokes"):
            if k in masks:
                union = cv2.bitwise_or(union, masks[k])
        return union
    return base


def make_28x28_from_box(masks: Dict[str, np.ndarray], box: Box, mask_name: str, padding: int = 5) -> Optional[np.ndarray]:
    mask = _mask_for_box(masks, mask_name)
    H, W = mask.shape[:2]
    x, y, w, h = box
    pad = max(padding, int(max(w, h) * 0.10))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
    crop = mask[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    pts = cv2.findNonZero(crop)
    if pts is None:
        return None
    rx, ry, rw, rh = cv2.boundingRect(pts)
    crop = crop[ry:ry + rh, rx:rx + rw]
    if crop.size == 0:
        return None
    # Light clean and thicken very thin strokes.
    crop = _clean_mask(crop, open_iter=0, close_iter=1, dilate_iter=0)
    ch, cw = crop.shape[:2]
    if ch <= 0 or cw <= 0:
        return None
    target = 20
    scale = target / float(max(ch, cw))
    new_w, new_h = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    xoff = (28 - new_w) // 2
    yoff = (28 - new_h) // 2
    canvas[yoff:yoff + new_h, xoff:xoff + new_w] = resized
    return canvas


def predict_canvas28(canvas28: np.ndarray, model: Any) -> Tuple[int, float, List[Dict[str, float]]]:
    shape = model.input_shape
    h, w, c = int(shape[1]), int(shape[2]), int(shape[3])
    img = cv2.resize(canvas28, (w, h), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
    if c == 1:
        x = img.reshape(1, h, w, 1)
    else:
        img3 = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB).astype("float32")
        # Match existing prediction_service RGB path.
        x = ((img3 / 127.5) - 1.0).reshape(1, h, w, 3)
    probs = model.predict(x, verbose=0)[0]
    digit = int(np.argmax(probs))
    conf = float(np.max(probs)) * 100.0
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [{"digit": int(i), "confidence": round(float(probs[i]) * 100.0, 2)} for i in top3_idx]
    return digit, conf, top3


def _group_results_rowwise(results: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not results:
        return []
    results = sorted(results, key=lambda r: r["box"][1])
    heights = [r["box"][3] for r in results]
    median_h = float(np.median(heights)) if heights else 10.0
    row_gap = max(8.0, median_h * 0.60)
    rows: List[List[Dict[str, Any]]] = []
    for r in results:
        x, y, w, h = r["box"]
        cy = y + h / 2
        placed = False
        for row in rows:
            row_cy = np.mean([a["box"][1] + a["box"][3] / 2 for a in row])
            if abs(cy - row_cy) <= row_gap:
                row.append(r)
                placed = True
                break
        if not placed:
            rows.append([r])
    for row in rows:
        row.sort(key=lambda r: r["box"][0])
    return rows


def _group_row_digits_into_numbers(row: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Groups adjacent digit boxes into multi-digit numbers using horizontal spacing.

    Example:
      [1] [2] [3] [4] [5]        -> 1 2 3 4 5
      [1][0]  [1][1]  [1][2]     -> 10 11 12
      [1][2][3][4][5]             -> 12345

    This does not change CNN prediction. It only formats already detected digits
    into number groups when the gap between two boxes is small.
    """
    if not row:
        return []

    row = sorted(row, key=lambda r: r["box"][0])
    if len(row) == 1:
        return [row]

    widths = [max(1, int(r["box"][2])) for r in row]
    heights = [max(1, int(r["box"][3])) for r in row]
    median_w = float(np.median(widths)) if widths else 10.0
    median_h = float(np.median(heights)) if heights else 20.0

    # Small gap => same number. Large gap => next number.
    # Works for 10, 11, 123, etc. while keeping spaced single digits separate.
    join_gap = max(5.0, min(median_w * 0.55, median_h * 0.35))

    groups: List[List[Dict[str, Any]]] = [[row[0]]]
    for prev, cur in zip(row, row[1:]):
        px, py, pw, ph = prev["box"]
        cx, cy, cw, ch = cur["box"]
        gap = cx - (px + pw)

        # If boxes touch/overlap or are very close, they belong to same multi-digit number.
        if gap <= join_gap:
            groups[-1].append(cur)
        else:
            groups.append([cur])

    return groups


def _format_number_rows(rows: List[List[Dict[str, Any]]], accepted_only: bool = True) -> Tuple[List[List[str]], str]:
    """Return number groups row-wise and a printable multi-line text output."""
    number_rows: List[List[str]] = []
    text_rows: List[str] = []

    for row in rows:
        number_groups = _group_row_digits_into_numbers(row)
        formatted_row: List[str] = []
        for group in number_groups:
            chars: List[str] = []
            for r in group:
                if r.get("digit") is None:
                    chars.append("?")
                elif accepted_only and r.get("status") != "ok":
                    chars.append("?")
                else:
                    chars.append(str(r.get("digit")))
            formatted_row.append("".join(chars))
        number_rows.append(formatted_row)
        text_rows.append("   ".join(formatted_row))

    return number_rows, "\n".join(text_rows)


def detect_digits_universal(
    img_rgb: np.ndarray,
    model: Optional[Any] = None,
    confidence_threshold: float = 20.0,
    max_digits: int = 150,
) -> Dict[str, Any]:
    img_rgb = _ensure_rgb(img_rgb)
    boxes, masks = find_universal_digit_boxes(img_rgb, max_digits=max_digits)
    annotated = img_rgb.copy()
    results: List[Dict[str, Any]] = []

    for pos, (box, mask_name, area) in enumerate(boxes, start=1):
        canvas28 = make_28x28_from_box(masks, box, mask_name)
        if canvas28 is None:
            continue
        if model is not None:
            digit, conf, top3 = predict_canvas28(canvas28, model)
            status = "ok" if conf >= confidence_threshold else "low_confidence"
            label = f"{digit} {conf:.0f}%"
        else:
            digit, conf, top3 = None, 0.0, []
            status = "detected"
            label = f"#{pos}"
        x, y, w, h = box
        color = (0, 190, 0) if status in {"ok", "detected"} else (255, 165, 0)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, max(1, int(max(img_rgb.shape[:2]) * 0.002)))
        cv2.putText(annotated, label, (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        results.append({
            "position": len(results) + 1,
            "digit": digit,
            "confidence": round(conf, 2),
            "status": status,
            "box": tuple(int(v) for v in box),
            "mask_name": mask_name,
            "canvas28": canvas28,
            "top3_predictions": top3,
        })

    rows = _group_results_rowwise(results)
    prediction_rows: List[str] = []
    raw_rows: List[str] = []
    for row in rows:
        prediction_rows.append("".join(str(r["digit"]) if r["digit"] is not None and r["status"] == "ok" else "?" for r in row))
        raw_rows.append("".join(str(r["digit"]) if r["digit"] is not None else "?" for r in row))

    # Number-wise grouping: keeps 10/11/123 as one number when digit boxes are close.
    number_rows, number_prediction = _format_number_rows(rows, accepted_only=True)
    raw_number_rows, raw_number_prediction = _format_number_rows(rows, accepted_only=False)

    # Combine masks for debug display.
    combined_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    for m in masks.values():
        combined_mask = cv2.bitwise_or(combined_mask, m)

    return {
        "success": len(results) > 0,
        "prediction": "\n".join(prediction_rows),
        "raw_prediction": "\n".join(raw_rows),
        "rows": rows,
        "digits": results,
        "digit_count": len(results),
        "annotated_image": annotated,
        "mask_image": combined_mask,
        "masks": masks,
        "message": "OK" if results else "No digit-like components detected.",
    }
