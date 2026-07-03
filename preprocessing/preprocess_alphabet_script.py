"""
preprocess_alphabet_script.py

Place:
    preprocessing/preprocess_alphabet_script.py

Preprocessing used by OCR inference model trained on EMNIST + MJSynth.
Model input:
    height = 32
    width  = 384
    channels = 1
"""

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps, ImageFilter


def to_pil_image(image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (str, Path)):
        return Image.open(image)
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    raise TypeError(f"Unsupported image type: {type(image)}")


def auto_make_dark_text_on_white(gray_img: Image.Image) -> Image.Image:
    img = gray_img.convert("L")
    arr = np.array(img)

    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if np.median(border) < 127:
        img = ImageOps.invert(img)
    return img


def crop_content(gray_img: Image.Image, padding: int = 2) -> Image.Image:
    img = gray_img.convert("L")
    arr = np.array(img)

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
        img = crop_content(img, padding=2)

    w, h = img.size
    scale = min(img_w / max(w, 1), img_h / max(h, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (img_w, img_h), 255)
    y = (img_h - new_h) // 2
    canvas.paste(img, (0, y))

    arr = np.array(canvas).astype("float32")
    arr = (255.0 - arr) / 255.0
    arr = np.expand_dims(arr, axis=-1)
    return arr.astype("float32")
