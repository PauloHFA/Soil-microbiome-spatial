"""
Evaluation package for balanced metrics, bootstrap confidence intervals, and hypothesis tests.
"""

from .metrics import compute_classification_metrics, compute_confusion_matrix_summary
from .bootstrap_ci import compute_spatial_bootstrap_ci, compute_spatial_unit_metric_ci
from .hypothesis_test import evaluate_multimarker_hypothesis, test_multimarker_null_hypothesis

__all__ = [
    "compute_classification_metrics",
    "compute_confusion_matrix_summary",
    "compute_spatial_bootstrap_ci",
    "compute_spatial_unit_metric_ci",
    "evaluate_multimarker_hypothesis",
    "test_multimarker_null_hypothesis",
]
