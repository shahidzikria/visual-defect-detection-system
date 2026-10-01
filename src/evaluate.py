"""
Evaluation module – computes test metrics, generates confusion matrix,
ROC curve, and performs error analysis on false positives / negatives.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from loguru import logger
from tqdm import tqdm

from src.config import CLASS_NAMES, OUTPUT_DIR, MODEL_DIR, BEST_MODEL_FILENAME
from src.model import DefectClassifier, load_model_for_inference


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Collect predictions
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model: DefectClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Run inference on the entire loader.

    Returns:
        all_labels   – ground-truth labels  (N,)
        all_preds    – predicted labels      (N,)
        all_probs    – softmax probabilities  (N, C)
        all_paths    – file paths for error analysis
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    all_paths = []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = probs.max(1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    # Gather file paths from the dataset
    dataset = loader.dataset
    if hasattr(dataset, "samples"):
        all_paths = [s[0] for s in dataset.samples]
    elif hasattr(dataset, "imgs"):
        all_paths = [s[0] for s in dataset.imgs]

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        all_paths,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Compute & log metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
) -> Dict[str, float]:
    """Return a dict of evaluation metrics."""
    # Use probability of the normal class (index 1) for ROC-AUC,
    # which aligns with sklearn's default pos_label=1.
    # Alternatively: roc_auc_score(y_true, y_probs[:, 0]) with pos_label=0
    # would give 1 - AUC, which was the bug.
    roc_auc = roc_auc_score(y_true, y_probs[:, 1])

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", pos_label=0),
        "recall": recall_score(y_true, y_pred, average="binary", pos_label=0),
        "f1_score": f1_score(y_true, y_pred, average="binary", pos_label=0),
        "roc_auc": roc_auc,
    }
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    """Save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved → {save_path}")


def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray, save_path: Path) -> None:
    """Save an ROC curve plot."""
    # Use probability of the normal class (index 1) aligned with sklearn default
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    auc_val = roc_auc_score(y_true, y_probs[:, 1])

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"ROC curve saved → {save_path}")


def plot_training_history(history_path: Path, save_path: Path) -> None:
    """Plot training / validation loss and accuracy curves."""
    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=4)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="s", markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o", markersize=4)
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", marker="s", markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Training curves saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Error analysis
# ──────────────────────────────────────────────────────────────────────────────

def error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    paths: List[str],
) -> Dict[str, list]:
    """
    Identify and log false positives and false negatives.
    Returns a dict with lists of misclassified samples.
    """
    errors = {"false_positives": [], "false_negatives": []}

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 0:
            # Normal classified as defective → False Positive
            errors["false_positives"].append({
                "path": paths[i] if i < len(paths) else f"sample_{i}",
                "true_label": CLASS_NAMES[y_true[i]],
                "pred_label": CLASS_NAMES[y_pred[i]],
                "confidence": float(y_probs[i].max()),
            })
        elif y_true[i] == 0 and y_pred[i] == 1:
            # Defective classified as normal → False Negative (more critical!)
            errors["false_negatives"].append({
                "path": paths[i] if i < len(paths) else f"sample_{i}",
                "true_label": CLASS_NAMES[y_true[i]],
                "pred_label": CLASS_NAMES[y_pred[i]],
                "confidence": float(y_probs[i].max()),
            })

    logger.info(
        f"Error analysis: {len(errors['false_positives'])} FPs, "
        f"{len(errors['false_negatives'])} FNs"
    )
    return errors


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Full evaluation pipeline
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: DefectClassifier,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict:
    """Run complete evaluation: metrics, plots, and error analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect predictions
    y_true, y_pred, y_probs, paths = collect_predictions(model, test_loader, device)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_probs)
    logger.info("=" * 60)
    logger.info("TEST SET EVALUATION RESULTS")
    logger.info("=" * 60)
    for k, v in metrics.items():
        logger.info(f"  {k:>12s}: {v:.4f}")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    logger.info(f"\n{report}")

    # Visualisations
    plot_confusion_matrix(y_true, y_pred, OUTPUT_DIR / "confusion_matrix.png")
    plot_roc_curve(y_true, y_probs, OUTPUT_DIR / "roc_curve.png")

    # Plot training curves if history exists
    history_path = OUTPUT_DIR / "training_history.json"
    if history_path.exists():
        plot_training_history(history_path, OUTPUT_DIR / "training_curves.png")

    # Error analysis
    errors = error_analysis(y_true, y_pred, y_probs, paths)

    # Measure inference latency
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    # Warm up
    for _ in range(10):
        _ = model(dummy)
    # Measure
    latencies = []
    for _ in range(50):
        start = time.perf_counter()
        _ = model(dummy)
        latencies.append((time.perf_counter() - start) * 1000)  # ms
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    logger.info(f"  Inference latency: avg={avg_latency:.1f}ms  p95={p95_latency:.1f}ms")

    # Save full report
    report_data = {
        "metrics": metrics,
        "classification_report": report,
        "error_analysis": {
            "num_false_positives": len(errors["false_positives"]),
            "num_false_negatives": len(errors["false_negatives"]),
            "false_positives": errors["false_positives"][:20],  # top 20
            "false_negatives": errors["false_negatives"][:20],
        },
        "inference_latency_ms": {
            "avg": round(avg_latency, 2),
            "p95": round(p95_latency, 2),
        },
    }

    report_path = OUTPUT_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)
    logger.info(f"Full evaluation report saved → {report_path}")

    return report_data
