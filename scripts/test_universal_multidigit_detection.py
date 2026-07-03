"""Test universal multi-digit detection without loading a CNN model.

Usage:
    python scripts/test_universal_multidigit_detection.py --image path/to/image.jpg --out debug_universal.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageOps
import numpy as np

from preprocessing.multidigit import detect_and_prepare_digits, draw_digit_boxes, auto_orient_for_multidigit_line


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--out", default="debug_universal_detection.png", help="Annotated output path")
    parser.add_argument("--mask-out", default=None, help="Mask output path")
    args = parser.parse_args()

    pil = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    image = np.array(pil)
    image = auto_orient_for_multidigit_line(image)

    candidates, mask, enhanced = detect_and_prepare_digits(image)
    fake_predictions = []
    for i, cand in enumerate(candidates, start=1):
        fake_predictions.append({"box": cand.box, "digit": i, "confidence": 100})

    annotated = draw_digit_boxes(image, fake_predictions)
    cv2.imwrite(args.out, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    mask_out = args.mask_out or str(Path(args.out).with_name(Path(args.out).stem + "_mask.png"))
    cv2.imwrite(mask_out, mask)

    print(f"Detected boxes: {len(candidates)}")
    print(f"Saved annotated image: {args.out}")
    print(f"Saved mask image: {mask_out}")
    for i, cand in enumerate(candidates, start=1):
        print(f"{i}: box={cand.box}")


if __name__ == "__main__":
    main()
