"""
multi_marker_fusion.py - Early (feature-level) and Late (decision-level) multi-marker integration.
"""

from typing import Tuple, Optional
import numpy as np


def early_fusion(x_16s: np.ndarray, x_its: np.ndarray) -> np.ndarray:
    """
    Early feature-level fusion by column-wise concatenation.
    Shape: (N, p_16s + p_its).
    """
    if x_16s.shape[0] != x_its.shape[0]:
        raise ValueError(
            f"Sample counts do not match between 16S ({x_16s.shape[0]}) and ITS ({x_its.shape[0]})"
        )
    return np.hstack([x_16s, x_its])


def late_fusion_predict(
    prob_16s: np.ndarray,
    prob_its: np.ndarray,
    weights: Tuple[float, float] = (0.5, 0.5)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Late decision-level fusion by weighted probability averaging.

    Args:
        prob_16s: (N, n_classes) predicted class probabilities from 16S model
        prob_its: (N, n_classes) predicted class probabilities from ITS model
        weights: (weight_16s, weight_its) blending weights summing to 1.0

    Returns:
        (predicted_classes, fused_probabilities)
    """
    w1, w2 = weights
    total_w = w1 + w2
    w1, w2 = w1 / total_w, w2 / total_w

    fused_prob = w1 * prob_16s + w2 * prob_its
    fused_preds = np.argmax(fused_prob, axis=1)
    return fused_preds, fused_prob
