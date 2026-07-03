"""
Dashboard chart data builder.
Returns JSON-serialisable dicts consumed by Chart.js on the frontend.
"""
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

MODELS_DIR = Path("models")
PLOTS_DIR  = Path("frontend/static/plots")


def load_training_metrics(model_name: str) -> Dict[str, Any]:
    """Load per-model metrics JSON saved by the training pipeline."""
    for suffix in ("_metrics.json", "_full_metrics.json"):
        p = MODELS_DIR / f"{model_name}{suffix}"
        if p.exists():
            return json.loads(p.read_text())
    return {}


def training_curves_data(model_name: str) -> Dict[str, Any]:
    """
    Returns paths to pre-generated training-curve PNGs.
    (Actual curve data would require saving history during training.)
    """
    return {
        "curve_img":     f"static/plots/{model_name}_training_curves.png",
        "cm_img":        f"static/plots/{model_name}_cm.png",
        "cm_norm_img":   f"static/plots/{model_name}_cm_norm.png",
        "roc_img":       f"static/plots/{model_name}_roc.png",
        "pr_curve_img":  f"static/plots/{model_name}_pr_curve.png",
        "per_class_img": f"static/plots/{model_name}_per_class.png",
        "cv_img":        f"static/plots/{model_name}_cv.png",
    }


def all_models_summary() -> List[Dict[str, Any]]:
    """Aggregate metrics for every trained model."""
    results = []
    for p in MODELS_DIR.glob("*_metrics.json"):
        if "adv" in p.stem or "distilled" in p.stem:
            continue
        data = json.loads(p.read_text())
        results.append({
            "model":        data.get("model", p.stem.replace("_metrics", "")),
            "accuracy":     data.get("accuracy", data.get("test_accuracy", 0)),
            "macro_f1":     data.get("macro_f1", 0),
            "auc_roc":      data.get("auc_roc", data.get("auc_roc_macro", 0)),
            "parameters":   data.get("parameters", 0),
            "training_time":data.get("training_time_seconds", data.get("training_time_s", 0)),
        })
    return sorted(results, key=lambda x: x["accuracy"], reverse=True)


def confusion_matrix_chartjs(cm_list: List[List[int]]) -> Dict[str, Any]:
    """Convert CM list-of-lists into Chart.js heatmap dataset."""
    cm = np.array(cm_list)
    datasets = []
    for i in range(10):
        for j in range(10):
            datasets.append({"x": j, "y": i, "v": int(cm[i, j])})
    return {
        "labels": list(range(10)),
        "data":   datasets,
        "max":    int(cm.max()),
    }


def robustness_chartjs() -> Dict[str, Any]:
    """Load FGSM robustness results if available."""
    results = {}
    for p in MODELS_DIR.glob("*_robustness.json"):
        data = json.loads(p.read_text())
        results[p.stem.replace("_robustness", "")] = data
    return results


def optimizer_comparison_chartjs() -> Dict[str, Any]:
    """Load optimizer experiment CSV as Chart.js friendly dict."""
    p = MODELS_DIR / "optimizer_experiments.csv"
    if not p.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(p)
    return {
        "labels":    df["optimizer"].tolist(),
        "accuracy":  df["test_accuracy"].round(4).tolist(),
        "macro_f1":  df["macro_f1"].round(4).tolist(),
    }
