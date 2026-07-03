import gzip
import re
import struct
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers


# ============================================================
# AUGMENTATION
# ============================================================

def build_augmentation_layer():
    return tf.keras.Sequential(
        [
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomContrast(0.10),
        ],
        name="digit_augmentation",
    )


# ============================================================
# COMMON IMAGE PREPROCESSING
# ============================================================

def _prepare_images(x, image_size=32, rgb=True):
    x = np.asarray(x)

    if x.ndim == 3:
        x = x[..., None]

    x = x.astype("float32")

    if x.max() > 1.5:
        x = x / 255.0

    if rgb:
        if x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)
    else:
        if x.shape[-1] == 3:
            x = np.mean(x, axis=-1, keepdims=True)

    if x.shape[1] != image_size or x.shape[2] != image_size:
        x = tf.image.resize(x, (image_size, image_size)).numpy()

    return x.astype("float32")


def _limit_data(x, y, limit=None, seed=42):
    if limit is None or len(x) <= limit:
        return x, y

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=limit, replace=False)
    return x[idx], y[idx]


# ============================================================
# MNIST DIRECT INTERNET LOAD
# ============================================================

def load_mnist(image_size=32, rgb=True, limit=None):
    print("Loading MNIST from TensorFlow internet download...")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = _prepare_images(x_train, image_size=image_size, rgb=rgb)
    x_test = _prepare_images(x_test, image_size=image_size, rgb=rgb)

    y_train = y_train.astype("int64")
    y_test = y_test.astype("int64")

    x_train, y_train = _limit_data(x_train, y_train, limit)
    x_test, y_test = _limit_data(x_test, y_test, limit)

    return (x_train, y_train), (x_test, y_test)


# ============================================================
# SVHN LOCAL .MAT LOAD
# ============================================================

def load_svhn(svhn_path="datasets", image_size=32, rgb=True, limit=None):
    print(f"Loading SVHN from: {svhn_path}")

    svhn_path = Path(svhn_path)

    train_file = svhn_path / "train_32x32.mat"
    test_file = svhn_path / "test_32x32.mat"

    if not train_file.exists():
        raise FileNotFoundError(f"SVHN train file not found: {train_file}")

    if not test_file.exists():
        raise FileNotFoundError(f"SVHN test file not found: {test_file}")

    train_mat = loadmat(train_file)
    test_mat = loadmat(test_file)

    x_train = train_mat["X"]
    y_train = train_mat["y"].reshape(-1)

    x_test = test_mat["X"]
    y_test = test_mat["y"].reshape(-1)

    # SVHN shape is H, W, C, N. Convert to N, H, W, C.
    x_train = np.transpose(x_train, (3, 0, 1, 2))
    x_test = np.transpose(x_test, (3, 0, 1, 2))

    # SVHN uses label 10 for digit 0.
    y_train = y_train % 10
    y_test = y_test % 10

    x_train = _prepare_images(x_train, image_size=image_size, rgb=rgb)
    x_test = _prepare_images(x_test, image_size=image_size, rgb=rgb)

    y_train = y_train.astype("int64")
    y_test = y_test.astype("int64")

    x_train, y_train = _limit_data(x_train, y_train, limit)
    x_test, y_test = _limit_data(x_test, y_test, limit)

    return (x_train, y_train), (x_test, y_test)


# ============================================================
# EMNIST IDX LOCAL LOAD
# ============================================================

def _find_emnist_file(folder, filename):
    folder = Path(folder)

    possible_files = [
        folder / filename,
        folder / f"{filename}.gz",
    ]

    for file in possible_files:
        if file.exists():
            return file

    raise FileNotFoundError(f"EMNIST file not found: {filename} in {folder}")


def _open_maybe_gzip(path):
    path = Path(path)

    with open(path, "rb") as f:
        magic = f.read(2)

    if magic == b"\x1f\x8b" or path.suffix == ".gz":
        return gzip.open(path, "rb")

    return open(path, "rb")


def _read_idx_images(path):
    with _open_maybe_gzip(path) as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))

        if magic != 2051:
            raise ValueError(f"Invalid image IDX file: {path}")

        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num, rows, cols)

    return images


def _read_idx_labels(path):
    with _open_maybe_gzip(path) as f:
        magic, num = struct.unpack(">II", f.read(8))

        if magic != 2049:
            raise ValueError(f"Invalid label IDX file: {path}")

        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels.astype("int64")


def load_emnist(
    emnist_path="datasets/EMNIST/gzip",
    subset="digits",
    image_size=32,
    rgb=True,
    limit=None,
    fix_orientation=True,
):
    print(f"Loading EMNIST {subset} from: {emnist_path}")

    train_img_file = _find_emnist_file(
        emnist_path,
        f"emnist-{subset}-train-images-idx3-ubyte",
    )
    train_lbl_file = _find_emnist_file(
        emnist_path,
        f"emnist-{subset}-train-labels-idx1-ubyte",
    )
    test_img_file = _find_emnist_file(
        emnist_path,
        f"emnist-{subset}-test-images-idx3-ubyte",
    )
    test_lbl_file = _find_emnist_file(
        emnist_path,
        f"emnist-{subset}-test-labels-idx1-ubyte",
    )

    x_train = _read_idx_images(train_img_file)
    y_train = _read_idx_labels(train_lbl_file)

    x_test = _read_idx_images(test_img_file)
    y_test = _read_idx_labels(test_lbl_file)

    # EMNIST images are usually rotated/transposed.
    if fix_orientation:
        x_train = np.transpose(x_train, (0, 2, 1))
        x_test = np.transpose(x_test, (0, 2, 1))

    # For digit recognition, keep only labels 0 to 9.
    train_mask = y_train <= 9
    test_mask = y_test <= 9

    x_train = x_train[train_mask]
    y_train = y_train[train_mask]

    x_test = x_test[test_mask]
    y_test = y_test[test_mask]

    x_train = _prepare_images(x_train, image_size=image_size, rgb=rgb)
    x_test = _prepare_images(x_test, image_size=image_size, rgb=rgb)

    y_train = y_train.astype("int64")
    y_test = y_test.astype("int64")

    x_train, y_train = _limit_data(x_train, y_train, limit)
    x_test, y_test = _limit_data(x_test, y_test, limit)

    return (x_train, y_train), (x_test, y_test)


# ============================================================
# MERGE MULTIPLE DATASETS
# ============================================================

def merge_datasets(dataset_list):
    x_train = np.concatenate([d[0][0] for d in dataset_list], axis=0)
    y_train = np.concatenate([d[0][1] for d in dataset_list], axis=0)

    x_test = np.concatenate([d[1][0] for d in dataset_list], axis=0)
    y_test = np.concatenate([d[1][1] for d in dataset_list], axis=0)

    idx_train = np.random.permutation(len(x_train))
    idx_test = np.random.permutation(len(x_test))

    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    x_test = x_test[idx_test]
    y_test = y_test[idx_test]

    return (x_train, y_train), (x_test, y_test)


# ============================================================
# OPTIONAL CUSTOM DATASET LOADER
# ============================================================

def _label_from_filename(filename):
    match = re.search(r"([0-9])", Path(filename).stem)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot find digit label in filename: {filename}")


def _load_images_from_folder(folder, image_size=32, rgb=True):
    folder = Path(folder)
    images = []
    labels = []

    extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]

    digit_folders = [p for p in folder.iterdir() if p.is_dir() and p.name.isdigit()]

    if digit_folders:
        for digit_folder in digit_folders:
            label = int(digit_folder.name)
            for ext in extensions:
                for img_path in digit_folder.glob(ext):
                    img = tf.keras.utils.load_img(
                        img_path,
                        color_mode="rgb" if rgb else "grayscale",
                        target_size=(image_size, image_size),
                    )
                    arr = tf.keras.utils.img_to_array(img)
                    images.append(arr)
                    labels.append(label)
    else:
        for ext in extensions:
            for img_path in folder.glob(ext):
                label = _label_from_filename(img_path.name)
                img = tf.keras.utils.load_img(
                    img_path,
                    color_mode="rgb" if rgb else "grayscale",
                    target_size=(image_size, image_size),
                )
                arr = tf.keras.utils.img_to_array(img)
                images.append(arr)
                labels.append(label)

    x = np.array(images, dtype="float32")
    y = np.array(labels, dtype="int64")

    x = _prepare_images(x, image_size=image_size, rgb=rgb)

    return x, y


def load_custom_dataset(dataset_path, image_size=32, rgb=True, test_size=0.2, limit=None):
    dataset_path = Path(dataset_path)

    train_dir = dataset_path / "train"
    test_dir = dataset_path / "test"

    if train_dir.exists() and test_dir.exists():
        x_train, y_train = _load_images_from_folder(train_dir, image_size=image_size, rgb=rgb)
        x_test, y_test = _load_images_from_folder(test_dir, image_size=image_size, rgb=rgb)
    else:
        x, y = _load_images_from_folder(dataset_path, image_size=image_size, rgb=rgb)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y,
        )

    x_train, y_train = _limit_data(x_train, y_train, limit)
    x_test, y_test = _limit_data(x_test, y_test, limit)

    return (x_train, y_train), (x_test, y_test)
