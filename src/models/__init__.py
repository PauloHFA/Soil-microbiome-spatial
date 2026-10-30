"""
Models package: feature representations, supervised classifiers, and multi-marker fusion.
"""

from .feature_builder import build_feature_sets
from .classifiers import get_classifier, train_classifier, evaluate_predictions
from .multi_marker_fusion import early_fusion, late_fusion_predict

__all__ = [
    "build_feature_sets",
    "get_classifier",
    "train_classifier",
    "evaluate_predictions",
    "early_fusion",
    "late_fusion_predict",
]
