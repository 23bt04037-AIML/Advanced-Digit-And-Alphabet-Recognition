"""
Evaluate every .keras model in the models/ directory on MNIST test set.
Reads cached _metrics.json if available, otherwise runs live inference.
Prints a clean summary table.
"""
import os, json, sys
sys.stdout.reconfigure(encoding="utf-8")  # ensure Unicode works on Windows cmd
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from pathlib import Path

tf.get_logger().setLevel("ERROR")

MODELS_DIR = Path("models")

# ── Load MNIST test set ────────────────────────────────────────────────────────
print("Loading MNIST test set …")
(_, _), (x_te_raw, y_te) = tf.keras.datasets.mnist.load_data()
# Grayscale normalised [0,1] for CNN models
x_te_grey = x_te_raw.astype("float32")[..., np.newaxis] / 255.0
# 32x32 RGB MobileNet-style for transfer models
x_te_rgb32 = np.repeat(x_te_raw[..., np.newaxis], 3, axis=-1).astype("float32")
x_te_rgb32 = tf.image.resize(x_te_rgb32, (32, 32)).numpy()
x_te_rgb32 = tf.keras.applications.mobilenet_v2.preprocess_input(x_te_rgb32)

print(f"Test samples: {len(y_te)}\n")

# ── Collect model files ────────────────────────────────────────────────────────
model_files = sorted(MODELS_DIR.glob("*.keras"))
# Exclude checkpoint variants (keep only the primary file per run)
# Keep *_best.keras only if no plain version exists
primary = {}
for p in model_files:
    stem = p.stem
    if stem.endswith("_best"):
        base = stem[:-5]
        if not (MODELS_DIR / f"{base}.keras").exists():
            primary[base] = p   # use _best as fallback
    else:
        primary[stem] = p       # plain file wins

rows = []

for stem, model_path in sorted(primary.items()):
    # Check for cached metrics
    metrics_path = MODELS_DIR / f"{stem}_metrics.json"
    cached = {}
    if metrics_path.exists():
        try:
            cached = json.loads(metrics_path.read_text())
        except Exception:
            pass

    # Determine if this is a transfer-learning model (3-channel input)
    is_transfer = ("mobilenetv2" in stem or "resnet50" in stem)

    # Load model
    print(f"  Loading {model_path.name} …", end=" ", flush=True)
    try:
        model = tf.keras.models.load_model(str(model_path))
    except Exception as e:
        print(f"FAILED ({e})")
        rows.append({"model": stem, "accuracy": "load error", "params": "—"})
        continue

    params = model.count_params()

    # Use cached accuracy if available (avoid re-running large models)
    acc_key = "test_accuracy" if "test_accuracy" in cached else "accuracy"
    if acc_key in cached:
        acc = cached[acc_key]
        print(f"cached  → acc={acc:.4f}")
    else:
        # Run live evaluation
        x_te = x_te_rgb32 if is_transfer else x_te_grey
        proba  = model.predict(x_te, verbose=0)
        y_pred = proba.argmax(axis=1)
        acc    = float(np.mean(y_pred == y_te))
        print(f"evaluated → acc={acc:.4f}")

    rows.append({
        "model":    stem,
        "accuracy": acc,
        "params":   params,
        "f1":       cached.get("macro_f1"),
        "auc_roc":  cached.get("auc_roc"),
    })

# ── Print table ───────────────────────────────────────────────────────────────
print()
print(f"{'Model':<30} {'Accuracy':>10} {'Macro F1':>10} {'AUC-ROC':>10} {'Params':>12}")
print("─" * 80)

rows_sorted = sorted(rows, key=lambda r: r["accuracy"] if isinstance(r["accuracy"], float) else 0, reverse=True)
for r in rows_sorted:
    acc  = f"{r['accuracy']:.4f}" if isinstance(r['accuracy'], float) else r['accuracy']
    f1   = f"{r['f1']:.4f}"  if r.get('f1')  else "  —   "
    auc  = f"{r['auc_roc']:.4f}" if r.get('auc_roc') else "  —   "
    par  = f"{r['params']:,}" if isinstance(r.get('params'), int) else "—"
    print(f"{r['model']:<30} {acc:>10} {f1:>10} {auc:>10} {par:>12}")

print()
best = max((r for r in rows if isinstance(r['accuracy'], float)), key=lambda r: r['accuracy'], default=None)
if best:
    print(f"🏆  Best model: {best['model']}  →  {best['accuracy']:.4f} accuracy")
