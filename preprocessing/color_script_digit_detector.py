import cv2
import numpy as np


# ------------------------------------------------------------
# 1) Create mask for colored / black / pencil / script digits
# ------------------------------------------------------------
def build_digit_mask(img_rgb):
    """
    Build a clean foreground mask for handwriting.

    Fix: do not use saturation alone for colored strokes. In phone photos the
    paper/background can also be saturated, so it becomes fake digits. We keep
    only pixels that are locally darker/lighter than the surrounding page.
    """
    if img_rgb is None:
        raise ValueError("Image is None")

    img_rgb = img_rgb.astype(np.uint8)
    if img_rgb.ndim == 2:
        rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    elif img_rgb.ndim == 3 and img_rgb.shape[2] == 4:
        rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
    else:
        rgb = img_rgb[:, :, :3]

    blur = cv2.GaussianBlur(rgb, (3, 3), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
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

    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    color_mask = ((s > 35) & (dark_diff > 8) & (v > 25)).astype(np.uint8) * 255

    dark_background = (float(np.mean(gray < 80)) > 0.20) or (float(np.mean(gray)) < 115)
    base = _mask_from_diff(light_diff if dark_background else dark_diff, 12.0)
    mask = cv2.bitwise_or(base, color_mask)

    kernel_small = np.ones((2, 2), np.uint8)
    kernel_close = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    mask = cv2.dilate(mask, kernel_small, iterations=1)

    # Remove photo/UI border strips.
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



def _keep_main_digit_rows_plain(boxes):
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
    row_h = [float(np.median([b[3] for b in row])) for row in rows]
    row_area = [float(np.median([b[2] * b[3] for b in row])) for row in rows]
    best_h = max(row_h)
    best_area = max(row_area)
    kept = []
    for row, mh, ma in zip(rows, row_h, row_area):
        if mh >= best_h * 0.52 and ma >= best_area * 0.18:
            kept.extend(row)
    return kept if kept else boxes

# ------------------------------------------------------------
# 2) Find digit boxes
# ------------------------------------------------------------
def find_digit_boxes(mask, max_digits=100):
    """
    Finds connected digit components and returns bounding boxes.
    """
    H, W = mask.shape[:2]
    img_area = H * W

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    boxes = []

    min_area = max(15, int(img_area * 0.00003))
    max_area = int(img_area * 0.20)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < min_area:
            continue
        if area > max_area:
            continue
        if w < 4 or h < 8:
            continue

        aspect = w / float(h)

        # Remove long horizontal/vertical noise
        if aspect > 3.5 and h < 25:
            continue
        if aspect < 0.08:
            continue

        boxes.append((int(x), int(y), int(w), int(h)))

    boxes = merge_close_boxes(boxes)

    # Remove very small boxes again after merge
    clean = []
    for x, y, w, h in boxes:
        if w >= 4 and h >= 8:
            clean.append((x, y, w, h))

    clean = _keep_main_digit_rows_plain(clean)
    clean = sort_boxes_rowwise(clean)
    return clean[:max_digits]


# ------------------------------------------------------------
# 3) Merge nearby parts of same digit
# ------------------------------------------------------------
def merge_close_boxes(boxes):
    """
    Merges components that belong to same digit.
    Useful for broken pencil/script strokes.
    """
    if not boxes:
        return []

    boxes = list(boxes)
    changed = True

    while changed:
        changed = False
        merged = []
        used = [False] * len(boxes)

        for i, b1 in enumerate(boxes):
            if used[i]:
                continue

            x1, y1, w1, h1 = b1
            bx1, by1, bx2, by2 = x1, y1, x1 + w1, y1 + h1

            for j, b2 in enumerate(boxes):
                if i == j or used[j]:
                    continue

                x2, y2, w2, h2 = b2
                cx1, cy1, cx2, cy2 = x2, y2, x2 + w2, y2 + h2

                # distance between boxes
                gap_x = max(0, max(bx1, cx1) - min(bx2, cx2))
                gap_y = max(0, max(by1, cy1) - min(by2, cy2))

                same_row = abs((by1 + by2) / 2 - (cy1 + cy2) / 2) < max(h1, h2) * 0.55

                # merge if close and likely part of same digit
                if same_row and gap_x < max(4, min(w1, w2) * 0.35) and gap_y < max(6, min(h1, h2) * 0.35):
                    bx1 = min(bx1, cx1)
                    by1 = min(by1, cy1)
                    bx2 = max(bx2, cx2)
                    by2 = max(by2, cy2)
                    used[j] = True
                    changed = True

            used[i] = True
            merged.append((bx1, by1, bx2 - bx1, by2 - by1))

        boxes = merged

    return boxes


# ------------------------------------------------------------
# 4) Sort boxes row-wise
# ------------------------------------------------------------
def sort_boxes_rowwise(boxes):
    """
    Sorts boxes top-to-bottom and left-to-right.
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[1])

    heights = [h for _, _, _, h in boxes]
    median_h = np.median(heights)
    row_gap = max(12, median_h * 0.65)

    rows = []

    for box in boxes:
        x, y, w, h = box
        cy = y + h / 2

        placed = False
        for row in rows:
            row_cy = np.mean([r[1] + r[3] / 2 for r in row])
            if abs(cy - row_cy) <= row_gap:
                row.append(box)
                placed = True
                break

        if not placed:
            rows.append([box])

    sorted_boxes = []
    for row in rows:
        row = sorted(row, key=lambda b: b[0])
        sorted_boxes.extend(row)

    return sorted_boxes


# ------------------------------------------------------------
# 5) Convert each detected digit to MNIST-like 28x28 image
# ------------------------------------------------------------
def make_28x28_digit(mask, box, padding=6):
    x, y, w, h = box
    H, W = mask.shape[:2]

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(W, x + w + padding)
    y2 = min(H, y + h + padding)

    crop = mask[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    # Find tight stroke bbox inside crop
    pts = cv2.findNonZero(crop)
    if pts is None:
        return None

    rx, ry, rw, rh = cv2.boundingRect(pts)
    crop = crop[ry:ry + rh, rx:rx + rw]

    # Resize while keeping aspect ratio
    ch, cw = crop.shape[:2]
    if ch == 0 or cw == 0:
        return None

    scale = 20.0 / max(ch, cw)
    new_w = max(1, int(cw * scale))
    new_h = max(1, int(ch * scale))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


# ------------------------------------------------------------
# 6) Predict one digit using CNN model
# ------------------------------------------------------------
def predict_digit(canvas28, model):
    """
    Works with CNN models expecting:
    - (28, 28, 1)
    - (28, 28, 3)
    """
    shape = model.input_shape
    h, w, c = int(shape[1]), int(shape[2]), int(shape[3])

    img = cv2.resize(canvas28, (w, h), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0

    if c == 1:
        x = img.reshape(1, h, w, 1)
    else:
        img3 = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        x = img3.astype("float32") / 255.0
        x = x.reshape(1, h, w, 3)

    probs = model.predict(x, verbose=0)[0]
    digit = int(np.argmax(probs))
    conf = float(np.max(probs)) * 100

    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [
        {"digit": int(i), "confidence": round(float(probs[i]) * 100, 2)}
        for i in top3_idx
    ]

    return digit, conf, top3


# ------------------------------------------------------------
# 7) Main function: detect + predict all digits
# ------------------------------------------------------------
def detect_digits_from_image(
    img_rgb,
    model=None,
    confidence_threshold=20,
    max_digits=100
):
    """
    img_rgb: RGB numpy image
    model: keras model. If None, only detection boxes are returned.
    """

    mask = build_digit_mask(img_rgb)
    boxes = find_digit_boxes(mask, max_digits=max_digits)

    annotated = img_rgb.copy()
    results = []

    for idx, box in enumerate(boxes, start=1):
        canvas28 = make_28x28_digit(mask, box)
        if canvas28 is None:
            continue

        x, y, w, h = box

        if model is not None:
            digit, conf, top3 = predict_digit(canvas28, model)
            status = "ok" if conf >= confidence_threshold else "low"
            label = f"{digit} {conf:.0f}%"
        else:
            digit, conf, top3 = None, 0, []
            status = "detected"
            label = f"#{idx}"

        color = (0, 200, 0) if status == "ok" or status == "detected" else (255, 165, 0)

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            annotated,
            label,
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        results.append({
            "position": idx,
            "digit": digit,
            "confidence": round(conf, 2),
            "status": status,
            "box": box,
            "canvas28": canvas28,
            "top3": top3
        })

    # Create row-wise output
    rows = group_results_rowwise(results)

    prediction_rows = []
    for row in rows:
        prediction_rows.append("".join(str(r["digit"]) if r["digit"] is not None else "?" for r in row))

    return {
        "success": len(results) > 0,
        "prediction": "\n".join(prediction_rows),
        "rows": rows,
        "digits": results,
        "digit_count": len(results),
        "mask": mask,
        "annotated": annotated
    }


# ------------------------------------------------------------
# 8) Group detected digits row-wise
# ------------------------------------------------------------
def group_results_rowwise(results):
    if not results:
        return []

    results = sorted(results, key=lambda r: r["box"][1])
    heights = [r["box"][3] for r in results]
    median_h = np.median(heights)
    row_gap = max(12, median_h * 0.65)

    rows = []

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