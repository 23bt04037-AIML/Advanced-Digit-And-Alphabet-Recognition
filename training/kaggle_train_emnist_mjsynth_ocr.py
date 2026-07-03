# ==========================================================
# KAGGLE OCR TRAINING: EMNIST + SVHN + SYNTHETIC MULTI-CHAR + MJSYNTH
# ONE MODEL: CRNN + CTC
# NO TRANSFER LEARNING: model trains from scratch.
# Learns single character, multi-digit, multi-alphabet, and mixed strings.
#
# Kaggle settings:
#   Internet: ON
#   Accelerator: GPU ON
#
# Output:
#   /kaggle/working/ocr_emnist_svhn_mjsynth_multichar/ocr_prediction_model.keras
#   /kaggle/working/ocr_emnist_svhn_mjsynth_multichar/charset.json
#   /kaggle/working/ocr_emnist_svhn_mjsynth_multichar_output.zip
# ==========================================================

# ==========================================================
# CELL 1: INSTALL
# ==========================================================
# Run this cell first. If Kaggle shows protobuf/tfds error, restart session once after this cell.

!pip install -q tensorflow-datasets==4.9.7 tensorflow-metadata==1.16.1 protobuf==5.29.5
!pip install -q datasets pillow tqdm pandas opencv-python

# ==========================================================
# CELL 2: IMPORTS + CONFIG
# ==========================================================

import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

print("TensorFlow:", tf.__version__)

# GPU memory growth
try:
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs:", gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print("GPU memory growth setup skipped:", e)

# ==========================================================
# NO OLD MODEL / NO TRANSFER LEARNING
# ==========================================================
# This notebook trains the CRNN+CTC OCR model from scratch.
# It does not load cnn_deep_nadam_best.keras or any old digit CNN model.

OUTPUT_DIR = Path("/kaggle/working/ocr_emnist_svhn_mjsynth_multichar")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Image size for CRNN OCR
IMG_H = 32
IMG_W = 384

# Dataset limits
EMNIST_LIMIT = 300000

# SVHN cropped digits from real-world street-view house numbers.
# Good for camera/photo-style digit recognition.
# Use None for full SVHN train+extra, but it will take longer.
SVHN_LIMIT = 200000

MJSYNTH_LIMIT = 100000

# Synthetic sequence limits. These are important for multi-digit and multi-alphabet OCR.
# SYNTH_TEXT_LIMIT: rendered typed strings like 12345, ABCD, a7B9.
# SYNTH_EMNIST_SEQ_LIMIT: stitched EMNIST handwritten characters into strings.
# SYNTH_SVHN_SEQ_LIMIT: stitched SVHN real-world digits into multi-digit strings.
SYNTH_TEXT_LIMIT = 80000
SYNTH_EMNIST_SEQ_LIMIT = 60000
SYNTH_SVHN_SEQ_LIMIT = 40000

# Training settings
BATCH_SIZE = 64
EPOCHS = 10

# For first test you can use:
# EMNIST_LIMIT = 5000
# SVHN_LIMIT = 5000
# MJSYNTH_LIMIT = 5000
# SYNTH_TEXT_LIMIT = 5000
# SYNTH_EMNIST_SEQ_LIMIT = 3000
# SYNTH_SVHN_SEQ_LIMIT = 3000
# EPOCHS = 3

MAX_LABEL_LEN = 48

# OCR charset
# This model can predict single characters and words.
CHARSET = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)

# If you want spaces/punctuation later, add:
# CHARSET = CHARSET + " .,;:!?'-/()&"

BLANK_INDEX = len(CHARSET)
NUM_CLASSES = len(CHARSET) + 1  # + CTC blank

char_to_num = {ch: i for i, ch in enumerate(CHARSET)}
num_to_char = {i: ch for ch, i in char_to_num.items()}

config = {
    "charset": CHARSET,
    "blank_index": BLANK_INDEX,
    "num_classes": NUM_CLASSES,
    "img_h": IMG_H,
    "img_w": IMG_W,
    "max_label_len": MAX_LABEL_LEN,
    "batch_size": BATCH_SIZE,
    "emnist_limit": EMNIST_LIMIT,
    "svhn_limit": SVHN_LIMIT,
    "mjsynth_limit": MJSYNTH_LIMIT,
    "synth_text_limit": SYNTH_TEXT_LIMIT,
    "synth_emnist_seq_limit": SYNTH_EMNIST_SEQ_LIMIT,
    "synth_svhn_seq_limit": SYNTH_SVHN_SEQ_LIMIT,
    "transfer_learning": False,
}

with open(OUTPUT_DIR / "charset.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)

print("Charset length:", len(CHARSET))
print("NUM_CLASSES with CTC blank:", NUM_CLASSES)
print("Output dir:", OUTPUT_DIR)

# ==========================================================
# CELL 3: PREPROCESSING FUNCTIONS
# ==========================================================

def to_pil_image(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image)
    return Image.open(image)


def clean_text(text):
    text = str(text)
    text = "".join(ch for ch in text if ch in char_to_num)
    return text


def encode_text(text):
    text = clean_text(text)

    if len(text) == 0:
        return None

    if len(text) > MAX_LABEL_LEN:
        return None

    label = np.zeros((MAX_LABEL_LEN,), dtype=np.int32)

    for i, ch in enumerate(text):
        label[i] = char_to_num[ch]

    label_length = np.array([len(text)], dtype=np.int32)

    return label, label_length, text


def fix_emnist_orientation(img):
    img = to_pil_image(img).convert("L")
    img = ImageOps.mirror(img.rotate(90, expand=True))
    return img


def auto_make_dark_text_on_white(gray_img):
    """
    Converts image to approx black text on white background.

    Works for:
    - EMNIST after inversion
    - MJSynth colored/gray word images
    """
    img = gray_img.convert("L")
    arr = np.array(img)

    h, w = arr.shape
    border = np.concatenate([
        arr[0, :],
        arr[-1, :],
        arr[:, 0],
        arr[:, -1],
    ])

    border_median = np.median(border)

    # If border/background is dark, invert to make background white.
    if border_median < 127:
        img = ImageOps.invert(img)

    return img


def crop_content(gray_img, padding=2):
    img = gray_img.convert("L")
    arr = np.array(img)

    # foreground = darker than near-white
    mask = arr < 245
    ys, xs = np.where(mask)

    if len(xs) < 5 or len(ys) < 5:
        return img

    left = max(0, int(xs.min()) - padding)
    right = min(img.width, int(xs.max()) + 1 + padding)
    top = max(0, int(ys.min()) - padding)
    bottom = min(img.height, int(ys.max()) + 1 + padding)

    return img.crop((left, top, right, bottom))


def preprocess_line_image(image, crop=True):
    """
    OCR input:
        shape = (32, 384, 1)
        text stroke = high value
        background = low value
    """
    img = to_pil_image(image).convert("L")
    img = img.filter(ImageFilter.MedianFilter(size=3))

    img = auto_make_dark_text_on_white(img)

    if crop:
        img = crop_content(img, padding=2)

    w, h = img.size
    scale = min(IMG_W / max(w, 1), IMG_H / max(h, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (IMG_W, IMG_H), 255)

    # Left aligned for CTC
    x = 0
    y = (IMG_H - new_h) // 2
    canvas.paste(img, (x, y))

    arr = np.array(canvas).astype("float32")

    # Convert to black background + white text/strokes
    arr = (255.0 - arr) / 255.0

    arr = np.expand_dims(arr, axis=-1)

    return arr.astype("float32")


def preprocess_emnist_for_ocr(example_image):
    """
    EMNIST image -> normal OCR line image.
    EMNIST is usually white char on black background.
    Convert to black char on white background first.
    """
    img = fix_emnist_orientation(example_image)
    img = ImageOps.invert(img.convert("L"))
    return preprocess_line_image(img, crop=True)


def preprocess_svhn_for_ocr(example_image):
    """
    SVHN cropped image -> OCR line image.

    TFDS svhn_cropped images are RGB cropped digit images.
    We convert them to grayscale and reuse the same OCR preprocessing.
    """
    img = to_pil_image(example_image).convert("L")
    return preprocess_line_image(img, crop=True)


def make_ctc_sample(image, text, time_steps):
    encoded = encode_text(text)

    if encoded is None:
        return None

    label, label_length, cleaned = encoded

    image_arr = preprocess_line_image(image, crop=True)

    x = {
        "image": image_arr,
        "label": label,
        "input_length": np.array([time_steps], dtype=np.int32),
        "label_length": label_length,
    }

    y = np.zeros((1,), dtype=np.float32)

    return x, y


def make_ctc_sample_from_preprocessed(image_arr, text, time_steps):
    encoded = encode_text(text)

    if encoded is None:
        return None

    label, label_length, cleaned = encoded

    x = {
        "image": image_arr,
        "label": label,
        "input_length": np.array([time_steps], dtype=np.int32),
        "label_length": label_length,
    }

    y = np.zeros((1,), dtype=np.float32)

    return x, y


# ==========================================================
# CELL 4: BUILD MODEL
# ==========================================================

def build_crnn_ctc_model():
    image_input = layers.Input(shape=(IMG_H, IMG_W, 1), name="image")

    # CNN encoder for OCR.
    # Trained from scratch, no old model weights are loaded.
    x = layers.Conv2D(32, 3, padding="same", use_bias=True, name="conv1")(image_input)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)

    x = layers.Conv2D(32, 3, padding="same", use_bias=True, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)  # 16 x 192

    x = layers.Conv2D(64, 3, padding="same", use_bias=True, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)

    x = layers.Conv2D(64, 3, padding="same", use_bias=True, name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)  # 8 x 96

    x = layers.Conv2D(128, 3, padding="same", use_bias=True, name="conv5")(x)
    x = layers.BatchNormalization(name="bn5")(x)
    x = layers.Activation("relu", name="relu5")(x)

    x = layers.Conv2D(128, 3, padding="same", use_bias=True, name="conv6")(x)
    x = layers.BatchNormalization(name="bn6")(x)
    x = layers.Activation("relu", name="relu6")(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), name="pool3")(x)  # 4 x 96

    x = layers.Conv2D(256, 3, padding="same", use_bias=True, name="conv7")(x)
    x = layers.BatchNormalization(name="bn7")(x)
    x = layers.Activation("relu", name="relu7")(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), name="pool4")(x)  # 2 x 96

    x = layers.Conv2D(256, 3, padding="same", activation="relu", name="conv8")(x)

    # Convert feature map to sequence:
    # batch, height, width, channels -> batch, width, height*channels
    shape = K.int_shape(x)
    print("CNN output shape:", shape)

    x = layers.Permute((2, 1, 3), name="permute_width_time")(x)

    shape = K.int_shape(x)
    time_steps = shape[1]
    feature_dim = shape[2] * shape[3]

    x = layers.Reshape((time_steps, feature_dim), name="sequence")(x)

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.25),
        name="bilstm1"
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.25),
        name="bilstm2"
    )(x)

    y_pred = layers.Dense(NUM_CLASSES, activation="softmax", name="ctc_softmax")(x)

    prediction_model = models.Model(image_input, y_pred, name="ocr_prediction_model")

    label_input = layers.Input(shape=(MAX_LABEL_LEN,), dtype="int32", name="label")
    input_length = layers.Input(shape=(1,), dtype="int32", name="input_length")
    label_length = layers.Input(shape=(1,), dtype="int32", name="label_length")

    def ctc_loss_func(args):
        labels_true, pred, in_len, lab_len = args
        return K.ctc_batch_cost(labels_true, pred, in_len, lab_len)

    loss_output = layers.Lambda(
        ctc_loss_func,
        output_shape=(1,),
        name="ctc_loss"
    )([label_input, y_pred, input_length, label_length])

    training_model = models.Model(
        inputs=[image_input, label_input, input_length, label_length],
        outputs=loss_output,
        name="ocr_training_model"
    )

    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=lambda y_true, y_pred: y_pred
    )

    return training_model, prediction_model, time_steps


training_model, prediction_model, TIME_STEPS = build_crnn_ctc_model()

print("TIME_STEPS:", TIME_STEPS)
prediction_model.summary()


# ==========================================================
# CELL 5: ONLINE DATASET GENERATORS
# ==========================================================

def iter_emnist(limit=None):
    """
    EMNIST ByClass full train dataset if limit=None.
    """
    import tensorflow_datasets as tfds

    emnist_chars = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

    print("Loading EMNIST ByClass...")

    try:
        ds = tfds.data_source("emnist/byclass", split="train")

        if limit is None:
            n = len(ds)
        else:
            n = min(limit, len(ds))

        print("Using EMNIST samples:", n)

        for i in range(n):
            example = ds[i]

            img = np.array(example["image"]).squeeze()
            label = int(example["label"])

            text = emnist_chars[label]

            image_arr = preprocess_emnist_for_ocr(img)
            sample = make_ctc_sample_from_preprocessed(image_arr, text, TIME_STEPS)

            if sample is not None:
                yield sample

    except Exception as e:
        print("tfds.data_source failed; trying tfds.load fallback.")
        print(e)

        ds = tfds.load("emnist/byclass", split="train", as_supervised=True)

        count = 0

        for img, label in tfds.as_numpy(ds):
            if limit is not None and count >= limit:
                break

            label = int(label)
            text = emnist_chars[label]

            image_arr = preprocess_emnist_for_ocr(np.array(img).squeeze())
            sample = make_ctc_sample_from_preprocessed(image_arr, text, TIME_STEPS)

            if sample is not None:
                yield sample
                count += 1

def iter_svhn(limit=None):
    """
    SVHN Cropped train+extra digits.

    Notes:
    - TFDS usually gives labels as 0-9.
    - Original SVHN sometimes stores digit 0 as label 10, so this code safely maps 10 -> 0.
    """
    import tensorflow_datasets as tfds

    print("Loading SVHN Cropped train+extra...")

    ds_train = tfds.load(
        "svhn_cropped",
        split="train",
        as_supervised=True,
        shuffle_files=True
    )

    ds_extra = tfds.load(
        "svhn_cropped",
        split="extra",
        as_supervised=True,
        shuffle_files=True
    )

    ds = ds_train.concatenate(ds_extra)

    count = 0

    for img, label in tfds.as_numpy(ds):
        if limit is not None and count >= limit:
            break

        digit = int(label)

        # Safety for original SVHN convention where 10 means digit 0.
        if digit == 10:
            digit = 0

        if digit < 0 or digit > 9:
            continue

        text = str(digit)

        image_arr = preprocess_svhn_for_ocr(np.array(img))
        sample = make_ctc_sample_from_preprocessed(image_arr, text, TIME_STEPS)

        if sample is not None:
            yield sample
            count += 1



# ==========================================================
# SYNTHETIC MULTI-CHAR GENERATORS
# ==========================================================
# Why this is added:
# - EMNIST and SVHN cropped are mostly single characters/digits.
# - CTC OCR needs sequence training for outputs like 12345, ABCD, hello, A7b9.
# - MJSynth gives word-level text, but synthetic strings make multi-digit/mixed OCR stronger.

EMNIST_CHAR_CACHE = None
SVHN_DIGIT_CACHE = None
FONT_CACHE = None


def random_training_text(min_len=2, max_len=18):
    mode = random.choice([
        "digits",
        "upper",
        "lower",
        "mixed_alnum",
        "word_like",
    ])

    if mode == "digits":
        chars = "0123456789"
        length = random.randint(min_len, min(max_len, 14))
    elif mode == "upper":
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        length = random.randint(min_len, min(max_len, 12))
    elif mode == "lower":
        chars = "abcdefghijklmnopqrstuvwxyz"
        length = random.randint(min_len, min(max_len, 12))
    elif mode == "word_like":
        first = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        rest_len = random.randint(1, min(max_len - 1, 11))
        rest = "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rest_len))
        return clean_text(first + rest)
    else:
        chars = CHARSET
        length = random.randint(min_len, max_len)

    return clean_text("".join(random.choice(chars) for _ in range(length)))


def get_available_fonts():
    global FONT_CACHE

    if FONT_CACHE is not None:
        return FONT_CACHE

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    ]

    fonts = []
    for path in candidates:
        if os.path.exists(path):
            fonts.append(path)

    FONT_CACHE = fonts
    print("Available fonts:", len(FONT_CACHE))
    return FONT_CACHE


def load_random_font(size):
    fonts = get_available_fonts()

    if fonts:
        try:
            return ImageFont.truetype(random.choice(fonts), size=size)
        except Exception:
            pass

    return ImageFont.load_default()


def add_light_noise(gray_img):
    arr = np.array(gray_img).astype(np.int16)

    if random.random() < 0.50:
        noise = np.random.normal(0, random.uniform(2, 10), arr.shape)
        arr = arr + noise

    if random.random() < 0.25:
        # light gray background variation
        arr = arr + random.randint(-8, 8)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")

    if random.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.6)))

    return img


def render_synthetic_text_image(text):
    text = clean_text(text)

    if not text:
        text = "123"

    font_size = random.randint(22, 34)
    font = load_random_font(font_size)

    dummy = Image.new("L", (10, 10), 255)
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)

    text_w = max(1, bbox[2] - bbox[0])
    text_h = max(1, bbox[3] - bbox[1])

    pad_x = random.randint(6, 18)
    pad_y = random.randint(4, 10)
    img_w = min(max(text_w + pad_x * 2, 40), IMG_W)
    img_h = max(text_h + pad_y * 2, IMG_H)

    bg = random.randint(238, 255)
    img = Image.new("L", (img_w, img_h), bg)
    draw = ImageDraw.Draw(img)

    ink = random.randint(0, 45)
    x = pad_x
    y = (img_h - text_h) // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=ink)

    if random.random() < 0.30:
        angle = random.uniform(-3.0, 3.0)
        img = img.rotate(angle, expand=True, fillcolor=255)

    img = add_light_noise(img)
    return img


def iter_synthetic_text(limit=None):
    print("Generating synthetic multi-character text samples...")

    count = 0
    while limit is None or count < limit:
        text = random_training_text(min_len=2, max_len=18)
        img = render_synthetic_text_image(text)
        sample = make_ctc_sample(img, text, TIME_STEPS)

        if sample is not None:
            yield sample
            count += 1


def emnist_char_pil(example_image):
    img = fix_emnist_orientation(example_image)
    img = ImageOps.invert(img.convert("L"))
    img = crop_content(img, padding=1)
    return img


def build_emnist_char_cache(samples_per_class=250):
    global EMNIST_CHAR_CACHE

    if EMNIST_CHAR_CACHE is not None:
        return EMNIST_CHAR_CACHE

    import tensorflow_datasets as tfds

    print("Building EMNIST character cache for handwritten multi-char strings...")

    emnist_chars = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    cache = {ch: [] for ch in emnist_chars}

    try:
        ds = tfds.data_source("emnist/byclass", split="train")
        n = len(ds)

        for i in range(n):
            example = ds[i]
            label = int(example["label"])

            if label < 0 or label >= len(emnist_chars):
                continue

            ch = emnist_chars[label]

            if len(cache[ch]) >= samples_per_class:
                if all(len(v) >= samples_per_class for v in cache.values()):
                    break
                continue

            img = np.array(example["image"]).squeeze()
            cache[ch].append(emnist_char_pil(img))

    except Exception as e:
        print("EMNIST data_source cache failed; trying tfds.load fallback.")
        print(e)
        ds = tfds.load("emnist/byclass", split="train", as_supervised=True)

        for img, label in tfds.as_numpy(ds):
            label = int(label)

            if label < 0 or label >= len(emnist_chars):
                continue

            ch = emnist_chars[label]

            if len(cache[ch]) < samples_per_class:
                cache[ch].append(emnist_char_pil(np.array(img).squeeze()))

            if all(len(v) >= samples_per_class for v in cache.values()):
                break

    # Remove empty classes just in case.
    cache = {ch: imgs for ch, imgs in cache.items() if len(imgs) > 0}
    print("Cached EMNIST classes:", len(cache))

    EMNIST_CHAR_CACHE = cache
    return EMNIST_CHAR_CACHE


def svhn_digit_pil(example_image):
    img = to_pil_image(example_image).convert("L")
    img = auto_make_dark_text_on_white(img)
    img = crop_content(img, padding=1)
    return img


def build_svhn_digit_cache(samples_per_digit=500):
    global SVHN_DIGIT_CACHE

    if SVHN_DIGIT_CACHE is not None:
        return SVHN_DIGIT_CACHE

    import tensorflow_datasets as tfds

    print("Building SVHN digit cache for real-world multi-digit strings...")

    cache = {str(i): [] for i in range(10)}

    ds_train = tfds.load("svhn_cropped", split="train", as_supervised=True, shuffle_files=True)
    ds_extra = tfds.load("svhn_cropped", split="extra", as_supervised=True, shuffle_files=True)
    ds = ds_train.concatenate(ds_extra)

    for img, label in tfds.as_numpy(ds):
        digit = int(label)
        if digit == 10:
            digit = 0

        if digit < 0 or digit > 9:
            continue

        ch = str(digit)

        if len(cache[ch]) < samples_per_digit:
            cache[ch].append(svhn_digit_pil(np.array(img)))

        if all(len(v) >= samples_per_digit for v in cache.values()):
            break

    cache = {ch: imgs for ch, imgs in cache.items() if len(imgs) > 0}
    print("Cached SVHN digit classes:", len(cache))

    SVHN_DIGIT_CACHE = cache
    return SVHN_DIGIT_CACHE


def stitch_character_images(text, cache, max_canvas_w=IMG_W):
    text = clean_text(text)

    if not text:
        return None

    char_imgs = []

    for ch in text:
        if ch not in cache or not cache[ch]:
            return None

        src = random.choice(cache[ch]).copy().convert("L")
        src = auto_make_dark_text_on_white(src)
        src = crop_content(src, padding=1)

        target_h = random.randint(22, 30)
        scale = target_h / max(src.height, 1)
        new_w = max(3, int(src.width * scale))
        src = src.resize((new_w, target_h), Image.Resampling.LANCZOS)
        char_imgs.append(src)

    spacing_values = [random.randint(1, 8) for _ in range(max(0, len(char_imgs) - 1))]
    total_w = sum(img.width for img in char_imgs) + sum(spacing_values) + random.randint(8, 20)
    total_h = max(img.height for img in char_imgs) + random.randint(4, 10)

    total_w = min(total_w, max_canvas_w)
    canvas = Image.new("L", (total_w, total_h), 255)

    x = random.randint(2, 8)
    for i, img in enumerate(char_imgs):
        y = random.randint(1, max(1, total_h - img.height - 1))
        if x + img.width >= total_w:
            break
        canvas.paste(img, (x, y))
        if i < len(spacing_values):
            x += img.width + spacing_values[i]

    if random.random() < 0.25:
        canvas = canvas.rotate(random.uniform(-2.0, 2.0), expand=True, fillcolor=255)

    canvas = add_light_noise(canvas)
    return canvas


def iter_synthetic_emnist_sequences(limit=None):
    print("Generating stitched EMNIST multi-character samples...")
    cache = build_emnist_char_cache(samples_per_class=250)

    count = 0
    while limit is None or count < limit:
        text = random_training_text(min_len=2, max_len=14)
        text = "".join(ch for ch in text if ch in cache)

        if len(text) < 2:
            continue

        img = stitch_character_images(text, cache)
        if img is None:
            continue

        sample = make_ctc_sample(img, text, TIME_STEPS)
        if sample is not None:
            yield sample
            count += 1


def iter_synthetic_svhn_sequences(limit=None):
    print("Generating stitched SVHN multi-digit samples...")
    cache = build_svhn_digit_cache(samples_per_digit=500)

    count = 0
    while limit is None or count < limit:
        length = random.randint(2, 10)
        text = "".join(random.choice("0123456789") for _ in range(length))
        img = stitch_character_images(text, cache)

        if img is None:
            continue

        sample = make_ctc_sample(img, text, TIME_STEPS)
        if sample is not None:
            yield sample
            count += 1

def extract_mjsynth_image_text(example):
    """
    Robust extraction because HF dataset field names can differ.
    """
    img = None
    text = None

    for key in ["image", "img"]:
        if key in example:
            img = example[key]
            break

    for key in ["text", "label", "word", "transcription"]:
        if key in example:
            text = example[key]
            break

    # Some datasets store metadata; try common fallback.
    if text is None and "file_name" in example:
        name = str(example["file_name"])
        parts = name.split("_")
        if len(parts) >= 2:
            text = parts[1]

    return img, text


def iter_mjsynth(limit=9000000):
    """
    MJSynth Text Recognition online streaming.
    If limit=9000000, it tries to use full MJSynth.
    """
    from datasets import load_dataset

    print("Streaming MJSynth from Hugging Face...")

    ds = load_dataset(
        "priyank-m/MJSynth_text_recognition",
        split="train",
        streaming=True
    )

    count = 0

    for example in ds:
        if limit is not None and count >= limit:
            break

        img, text = extract_mjsynth_image_text(example)

        if img is None or text is None:
            continue

        text = clean_text(text)

        if len(text) == 0 or len(text) > MAX_LABEL_LEN:
            continue

        sample = make_ctc_sample(img, text, TIME_STEPS)

        if sample is not None:
            yield sample
            count += 1

def combined_generator():
    """
    One training stream:
    1. EMNIST single handwritten characters
    2. SVHN single real-world digits
    3. Synthetic rendered multi-character strings
    4. Stitched EMNIST handwritten multi-character strings
    5. Stitched SVHN real-world multi-digit strings
    6. MJSynth word samples
    """
    for sample in iter_emnist(EMNIST_LIMIT):
        yield sample

    for sample in iter_svhn(SVHN_LIMIT):
        yield sample

    for sample in iter_synthetic_text(SYNTH_TEXT_LIMIT):
        yield sample

    for sample in iter_synthetic_emnist_sequences(SYNTH_EMNIST_SEQ_LIMIT):
        yield sample

    for sample in iter_synthetic_svhn_sequences(SYNTH_SVHN_SEQ_LIMIT):
        yield sample

    for sample in iter_mjsynth(MJSYNTH_LIMIT):
        yield sample


# ==========================================================
# CELL 7: CREATE TF.DATA PIPELINE
# ==========================================================
import tensorflow_datasets as tfds

# Full EMNIST train count
if EMNIST_LIMIT is None:
    emnist_count = len(tfds.data_source("emnist/byclass", split="train"))
else:
    emnist_count = EMNIST_LIMIT

# SVHN train+extra count
if SVHN_LIMIT is None:
    try:
        svhn_builder = tfds.builder("svhn_cropped")
        svhn_count = (
            svhn_builder.info.splits["train"].num_examples
            + svhn_builder.info.splits["extra"].num_examples
        )
    except Exception:
        # Known SVHN cropped train+extra total fallback.
        svhn_count = 73257 + 531131
else:
    svhn_count = SVHN_LIMIT

mjsynth_count = MJSYNTH_LIMIT
synth_text_count = SYNTH_TEXT_LIMIT
synth_emnist_seq_count = SYNTH_EMNIST_SEQ_LIMIT
synth_svhn_seq_count = SYNTH_SVHN_SEQ_LIMIT

TOTAL_SAMPLES = (
    emnist_count
    + svhn_count
    + synth_text_count
    + synth_emnist_seq_count
    + synth_svhn_seq_count
    + mjsynth_count
)
STEPS_PER_EPOCH = max(1, TOTAL_SAMPLES // BATCH_SIZE)

print("EMNIST single-char samples:", emnist_count)
print("SVHN single-digit samples:", svhn_count)
print("Synthetic rendered multi-char samples:", synth_text_count)
print("Synthetic EMNIST multi-char samples:", synth_emnist_seq_count)
print("Synthetic SVHN multi-digit samples:", synth_svhn_seq_count)
print("MJSynth word samples:", mjsynth_count)
print("Total samples per epoch:", TOTAL_SAMPLES)
print("Batch size:", BATCH_SIZE)
print("Steps per epoch:", STEPS_PER_EPOCH)

output_signature = (
    {
        "image": tf.TensorSpec(shape=(IMG_H, IMG_W, 1), dtype=tf.float32),
        "label": tf.TensorSpec(shape=(MAX_LABEL_LEN,), dtype=tf.int32),
        "input_length": tf.TensorSpec(shape=(1,), dtype=tf.int32),
        "label_length": tf.TensorSpec(shape=(1,), dtype=tf.int32),
    },
    tf.TensorSpec(shape=(1,), dtype=tf.float32),
)

train_ds = tf.data.Dataset.from_generator(
    combined_generator,
    output_signature=output_signature
)

train_ds = train_ds.shuffle(2048)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
train_ds = train_ds.repeat()


# ==========================================================
# CELL 8: TRAIN
# ==========================================================

callbacks = [
    ModelCheckpoint(
        filepath=str(OUTPUT_DIR / "best_ocr_training_model.keras"),
        monitor="loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    EarlyStopping(
        monitor="loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    CSVLogger(str(OUTPUT_DIR / "training_log.csv"))
]

history = training_model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=callbacks
)

# Save prediction model for Streamlit/inference
prediction_model.save(OUTPUT_DIR / "ocr_prediction_model.keras")

pd.DataFrame(history.history).to_csv(
    OUTPUT_DIR / "training_history.csv",
    index=False
)

print("Training complete.")
print("Prediction model saved:", OUTPUT_DIR / "ocr_prediction_model.keras")
print("Charset saved:", OUTPUT_DIR / "charset.json")


# ==========================================================
# CELL 9: TEST PREDICTION
# ==========================================================

def decode_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    decoded = K.ctc_decode(
        pred,
        input_length=input_len,
        greedy=True
    )[0][0].numpy()

    results = []

    for row in decoded:
        text = ""

        for idx in row:
            idx = int(idx)

            if idx == -1 or idx == BLANK_INDEX:
                continue

            if idx in num_to_char:
                text += num_to_char[idx]

        results.append(text)

    return results


def predict_text(image):
    arr = preprocess_line_image(image, crop=True)
    arr = np.expand_dims(arr, axis=0)

    pred = prediction_model.predict(arr, verbose=0)
    return decode_predictions(pred)[0]


# Test on 10 EMNIST samples
print("\nTesting on EMNIST samples:")
import tensorflow_datasets as tfds

try:
    emnist_test = tfds.data_source("emnist/byclass", split="test")
    emnist_chars = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

    for i in range(10):
        example = emnist_test[i]
        raw_img = np.array(example["image"]).squeeze()
        true_label = int(example["label"])
        true_text = emnist_chars[true_label]

        img = fix_emnist_orientation(raw_img)
        img = ImageOps.invert(img.convert("L"))

        pred_text = predict_text(img)

        print(f"TRUE: {true_text} | PRED: {pred_text}")

except Exception as e:
    print("EMNIST test preview skipped:", e)


# Test on 10 SVHN samples
print("\nTesting on SVHN samples:")

try:
    svhn_test = tfds.load("svhn_cropped", split="test", as_supervised=True)

    c = 0
    for raw_img, true_label in tfds.as_numpy(svhn_test):
        digit = int(true_label)
        if digit == 10:
            digit = 0

        true_text = str(digit)

        img = to_pil_image(raw_img).convert("L")
        pred_text = predict_text(img)

        print(f"TRUE: {true_text} | PRED: {pred_text}")

        c += 1
        if c >= 10:
            break

except Exception as e:
    print("SVHN test preview skipped:", e)


# Test on 5 MJSynth samples
print("\nTesting on MJSynth samples:")

try:
    from datasets import load_dataset

    mjs = load_dataset(
        "priyank-m/MJSynth_text_recognition",
        split="train",
        streaming=True
    )

    c = 0
    for example in mjs:
        img, text = extract_mjsynth_image_text(example)
        if img is None or text is None:
            continue

        true_text = clean_text(text)
        if not true_text:
            continue

        pred_text = predict_text(img)
        print(f"TRUE: {true_text} | PRED: {pred_text}")

        c += 1
        if c >= 5:
            break

except Exception as e:
    print("MJSynth test preview skipped:", e)




# Test on synthetic multi-character samples
print("\nTesting on synthetic multi-character samples:")

try:
    test_texts = ["12345", "98710", "ABC", "hello", "A7b9", "DigitOCR2026","3TANMAY9"]

    for true_text in test_texts:
        true_text = clean_text(true_text)
        img = render_synthetic_text_image(true_text)
        pred_text = predict_text(img)
        print(f"TRUE: {true_text} | PRED: {pred_text}")

except Exception as e:
    print("Synthetic multi-character test preview skipped:", e)

# ==========================================================
# CELL 10: ZIP OUTPUT
# ==========================================================

import shutil

zip_base = "/kaggle/working/ocr_emnist_svhn_mjsynth_multichar_output"
shutil.make_archive(zip_base, "zip", OUTPUT_DIR)

print("ZIP saved:", zip_base + ".zip")
print("Download from Kaggle Output panel.")
