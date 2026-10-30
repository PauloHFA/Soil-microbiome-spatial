"""
metrics.py - Balanced accuracy, Macro-F1, and confusion matrix analytics.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute balanced evaluation metrics suited for class-imbalanced ecological categories.

    Returns:
        Dict with 'balanced_accuracy', 'macro_f1', and 'overall_accuracy'
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    bacc = float(balanced_accuracy_score(y_t, y_p))
    mf1 = float(f1_score(y_t, y_p, average="macro", zero_division=0.0))
    acc = float(np.mean(y_t == y_p)) if len(y_t) > 0 else 0.0

    return {
        "balanced_accuracy": bacc,
        "macro_f1": mf1,
        "overall_accuracy": acc,
    }


def compute_confusion_matrix_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Optional[List[str]] = None,
    label_ids: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Compute raw confusion matrix and per-class true/false positive counts.
    """
    if label_ids is not None:
        cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    else:
        cm = confusion_matrix(y_true, y_pred)
    classes = class_labels if class_labels else [str(c) for c in np.unique(y_true)]

    return {
        "matrix": cm.tolist(),
        "classes": classes,
        "total_samples": int(len(y_true)),
    }
