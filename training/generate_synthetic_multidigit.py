"""
Generate synthetic multi-digit images from MNIST.

Use this to test your multi-digit detector with known labels.

Run:
    python training/generate_synthetic_multidigit.py --out dataset/synthetic_multidigit --count 3000

Output:
    dataset/synthetic_multidigit/images/*.png
    dataset/synthetic_multidigit/labels.csv

This does NOT replace real handwritten/camera data, but it helps you debug
segmentation and final number accuracy.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.datasets import mnist


def _tight_crop(img: np.ndarray) -> np.ndarray:
    ys, xs = np.where(img > 15)
    if len(xs) == 0 or len(ys) == 0:
        return img
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _augment_digit(img: np.ndarray) -> np.ndarray:
    img = _tight_crop(img)

    scale = random.uniform(0.85, 1.35)
    h, w = img.shape
    img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    angle = random.uniform(-18, 18)
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, m, (w, h), borderValue=0)

    if random.random() < 0.35:
        k = np.ones((2, 2), np.uint8)
        img = cv2.dilate(img, k, iterations=1)
    if random.random() < 0.25:
        k = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, k, iterations=1)

    return img


def generate(out_dir: Path, count: int, min_len: int = 2, max_len: int = 6, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    (x_train, y_train), _ = mnist.load_data()
    by_digit = {d: np.where(y_train == d)[0].tolist() for d in range(10)}

    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "labels.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "boxes"])
        writer.writeheader()

        for i in range(count):
            n_digits = random.randint(min_len, max_len)
            label_digits = [random.randint(0, 9) for _ in range(n_digits)]

            bg_value = random.choice([0, 255, random.randint(180, 245)])
            digit_white = bg_value < 128
            canvas_h = random.randint(48, 90)
            canvas_w = random.randint(140, 320)
            canvas = np.full((canvas_h, canvas_w), bg_value, dtype=np.uint8)

            x_cursor = random.randint(5, 15)
            boxes = []
            for d in label_digits:
                idx = random.choice(by_digit[d])
                digit = _augment_digit(x_train[idx])
                if not digit_white:
                    # black digit on light page
                    digit = 255 - digit
                    fg_mask = digit < 240
                else:
                    fg_mask = digit > 15

                dh, dw = digit.shape
                if x_cursor + dw + 5 >= canvas_w:
                    break
                y = random.randint(5, max(5, canvas_h - dh - 5))

                roi = canvas[y:y + dh, x_cursor:x_cursor + dw]
                if digit_white:
                    roi[fg_mask] = np.maximum(roi[fg_mask], digit[fg_mask])
                else:
                    roi[fg_mask] = np.minimum(roi[fg_mask], digit[fg_mask])

                boxes.append({"digit": d, "x": x_cursor, "y": y, "w": dw, "h": dh})
                x_cursor += dw + random.randint(2, 18)

            if not boxes:
                continue

            label = "".join(str(b["digit"]) for b in boxes)

            # Add noise / blur / shadows.
            if random.random() < 0.55:
                noise = np.random.normal(0, random.uniform(3, 12), canvas.shape)
                canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            if random.random() < 0.25:
                canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

            filename = f"multi_{i:05d}_{label}.png"
            cv2.imwrite(str(image_dir / filename), canvas)
            writer.writerow({"filename": filename, "label": label, "boxes": boxes})

    print(f"Saved {count} synthetic multi-digit images to {out_dir}")
    print(f"Labels: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dataset/synthetic_multidigit", help="Output folder")
    parser.add_argument("--count", type=int, default=3000, help="Number of images")
    parser.add_argument("--min-len", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(Path(args.out), args.count, args.min_len, args.max_len, args.seed)
