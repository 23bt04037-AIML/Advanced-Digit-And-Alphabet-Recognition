"""
Command-line test for one multi-digit image.

Run from project root:
    python scripts/test_multidigit_image.py --image path/to/number.png --model cnn_deep
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.prediction_service import multi_digit_predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="cnn_deep")
    parser.add_argument("--conf", type=float, default=50.0)
    parser.add_argument("--out", default="outputs/multidigit_debug.png")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    # Convert BGR to RGB because the drawing function uses RGB-style arrays.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = multi_digit_predict(img_rgb, model_name=args.model, confidence_threshold=args.conf)
    print("Prediction:", result.get("prediction"))
    print("Raw prediction:", result.get("raw_prediction"))
    print("Message:", result.get("message"))

    for d in result.get("digits", []):
        print(
            f"#{d['position']}: digit={d['digit']} conf={d['confidence']} "
            f"status={d['status']} box={d['box']}"
        )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = result.get("annotated_image")
    if annotated is not None:
        cv2.imwrite(str(out_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print("Saved annotated image:", out_path)
