"""
Comprehensive evaluation metrics module.
Computes accuracy, precision, recall, F1, AUC-ROC, confusion matrix,
per-class stats, and generates all evaluation plots.
"""
import json, logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize

logger    = logging.getLogger(__name__)
PLOTS_DIR = Path("frontend/static/plots"); PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR= Path("models"); MODELS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE METRICS
# ═══════════════════════════════════════════════════════════════════════════════
def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        y_proba: np.ndarray, num_classes: int = 10) -> Dict[str, Any]:
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    metrics: Dict[str, Any] = {
        "accuracy":            round(accuracy_score(y_true, y_pred), 6),
        "macro_precision":     round(precision_score(y_true, y_pred, average="macro",   zero_division=0), 6),
        "macro_recall":        round(recall_score(   y_true, y_pred, average="macro",   zero_division=0), 6),
        "macro_f1":            round(f1_score(       y_true, y_pred, average="macro",   zero_division=0), 6),
        "weighted_f1":         round(f1_score(       y_true, y_pred, average="weighted",zero_division=0), 6),
        "auc_roc_macro":       round(roc_auc_score(  y_bin,  y_proba, multi_class="ovr",average="macro"), 6),
        "confusion_matrix":    confusion_matrix(y_true, y_pred).tolist(),
        "per_class": {
            str(c): {
                "precision": round(precision_score(y_true, y_pred, labels=[c], average="micro", zero_division=0), 4),
                "recall":    round(recall_score(   y_true, y_pred, labels=[c], average="micro", zero_division=0), 4),
                "f1":        round(f1_score(       y_true, y_pred, labels=[c], average="micro", zero_division=0), 4),
                "support":   int((y_true == c).sum()),
            }
            for c in range(num_classes)
        },
    }
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(cm: np.ndarray, model_name: str, normalize: bool = False):
    if normalize:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt, title_suffix = ".2f", " (Normalised)"
    else:
        cm_plot, fmt, title_suffix = cm, "d", ""

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=range(10), yticklabels=range(10),
                ax=ax, linewidths=0.4, cbar_kws={"shrink": 0.8})
    ax.set(title=f"{model_name} – Confusion Matrix{title_suffix}",
           xlabel="Predicted Label", ylabel="True Label")
    plt.tight_layout()
    suffix = "_norm" if normalize else ""
    out = PLOTS_DIR / f"{model_name}_cm{suffix}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray,
                    model_name: str, num_classes: int = 10):
    y_bin  = label_binarize(y_true, classes=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    for c, col in zip(range(num_classes), colors):
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_proba[:, c])
        auc = roc_auc_score(y_bin[:, c], y_proba[:, c])
        ax.plot(fpr, tpr, lw=1.5, color=col, label=f"Digit {c} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(title=f"{model_name} – ROC Curves (One-vs-Rest)",
           xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / f"{model_name}_roc.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_precision_recall_curves(y_true: np.ndarray, y_proba: np.ndarray,
                                  model_name: str, num_classes: int = 10):
    y_bin  = label_binarize(y_true, classes=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    for c, col in zip(range(num_classes), colors):
        p, r, _ = precision_recall_curve(y_bin[:, c], y_proba[:, c])
        ap = average_precision_score(y_bin[:, c], y_proba[:, c])
        ax.plot(r, p, lw=1.5, color=col, label=f"Digit {c} (AP={ap:.3f})")
    ax.set(title=f"{model_name} – Precision-Recall Curves",
           xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.05))
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / f"{model_name}_pr_curve.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_per_class_metrics(metrics: Dict[str, Any], model_name: str):
    classes = [str(i) for i in range(10)]
    prec = [metrics["per_class"][c]["precision"] for c in classes]
    rec  = [metrics["per_class"][c]["recall"]    for c in classes]
    f1   = [metrics["per_class"][c]["f1"]        for c in classes]

    x = np.arange(10); w = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, prec, w, label="Precision", color="#4e79a7")
    ax.bar(x,     rec,  w, label="Recall",    color="#f28e2b")
    ax.bar(x + w, f1,   w, label="F1",        color="#59a14f")
    ax.set(title=f"{model_name} – Per-Class Metrics", xticks=x, xticklabels=classes,
           xlabel="Digit Class", ylabel="Score", ylim=(0, 1.05))
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = PLOTS_DIR / f"{model_name}_per_class.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_model_comparison(summaries: List[Dict[str, Any]]):
    """Bar chart comparing all trained models on key metrics."""
    if not summaries:
        return
    df = pd.DataFrame(summaries)
    metrics_cols = ["accuracy", "macro_f1", "auc_roc_macro"]
    available    = [c for c in metrics_cols if c in df.columns]
    x = np.arange(len(df)); w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]
    for i, (col, color) in enumerate(zip(available, colors)):
        ax.bar(x + i * w, df[col], w, label=col, color=color)
    ax.set(title="Model Comparison", xticks=x + w,
           xticklabels=df.get("model", df.index), xlabel="Model", ylabel="Score")
    ax.legend(); ax.grid(alpha=0.3, axis="y"); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    out = PLOTS_DIR / "model_comparison.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL RUNNER
# ═══════════════════════════════════════════════════════════════════════════════
def full_evaluation(model, model_name: str, x_test, y_test) -> Dict[str, Any]:
    """Run predictions, compute all metrics, save all plots, return metrics dict."""
    import tensorflow as tf
    y_proba = model.predict(x_test, verbose=0)
    y_pred  = y_proba.argmax(axis=1)

    metrics = compute_all_metrics(y_test, y_pred, y_proba)
    metrics["model"] = model_name

    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, model_name, normalize=False)
    plot_confusion_matrix(cm, model_name, normalize=True)
    plot_roc_curves(y_test, y_proba, model_name)
    plot_precision_recall_curves(y_test, y_proba, model_name)
    plot_per_class_metrics(metrics, model_name)

    out_path = MODELS_DIR / f"{model_name}_full_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Saved full metrics to {out_path}")
    return metrics
