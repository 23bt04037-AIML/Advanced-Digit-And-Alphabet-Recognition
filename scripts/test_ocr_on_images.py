"""Small local test script.
Run from project root after replacing ocr_prediction_service.py:
    python scripts/test_ocr_on_images.py path/to/image.png
"""
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.ocr_prediction_service import predict_full_image_ocr

for arg in sys.argv[1:]:
    img = Image.open(arg)
    result = predict_full_image_ocr(img, mode="auto", debug=True)
    print("\nIMAGE:", arg)
    print("MODE:", result.get("segmentation_mode"))
    print("TEXT:")
    print(result.get("text", ""))
    print("CONF:", result.get("confidence"))
    print("DEBUG:", {k:v for k,v in result.items() if k.startswith('debug_')})
