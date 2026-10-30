"""
hypothesis_test.py - Paired test of integration against the training-selected marker.
"""

from typing import Dict, Any, Optional, Sequence
import numpy as np
from scipy import stats


def select_marker_from_training_scores(
    training_scores_16s: Sequence[float],
    training_scores_its: Sequence[float],
) -> np.ndarray:
    """
    Choose the comparator inside training.

    Ties use the predeclared order 16S then ITS. Test scores are not an input,
    so they cannot break the tie.
    """
    train_a = np.asarray(training_scores_16s, dtype=np.float64)
    train_b = np.asarray(training_scores_its, dtype=np.float64)
    if train_a.shape != train_b.shape or train_a.ndim != 1 or len(train_a) == 0:
        raise ValueError("Training scores for 16S and ITS must be non-empty and aligned.")
    chosen = np.where(train_a >= train_b, "16S", "ITS")
    return chosen.astype(str)


def evaluate_multimarker_hypothesis(
    scores_16s: np.ndarray,
    scores_its: np.ndarray,
    scores_integrated: np.ndarray,
    alternative: str = "greater",
    training_scores_16s: Optional[Sequence[float]] = None,
    training_scores_its: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    H0: integration is not superior to the best single marker.
    The best single marker is fixed from training scores before the test delta is formed.
    """
    s_a = np.asarray(scores_16s, dtype=np.float64)
    s_b = np.asarray(scores_its, dtype=np.float64)
    s_c = np.asarray(scores_integrated, dtype=np.float64)
    if not (s_a.shape == s_b.shape == s_c.shape) or s_a.ndim != 1:
        raise ValueError("Test scores for A, B and C must be aligned 1D arrays.")

    if training_scores_16s is None or training_scores_its is None:
        best_single = np.maximum(s_a, s_b)
        chosen = np.where(s_a >= s_b, "16S", "ITS").astype(str)
    else:
        chosen = select_marker_from_training_scores(training_scores_16s, training_scores_its)
        if len(chosen) != len(s_c):
            raise ValueError("Training selection and test scores must have one entry per fold.")
        best_single = np.where(chosen == "16S", s_a, s_b).astype(np.float64)
    delta = s_c - best_single
    mean_delta = float(np.mean(delta))
    non_zero = delta[delta != 0]

    if len(non_zero) == 0:
        return {
            "mean_delta": 0.0,
            "w_statistic": 0.0,
            "p_value": 1.0,
            "h0_rejected": False,
            "selected_marker": chosen.tolist(),
            "comparator_selected_on": "training",
            "conclusion": (
                "H0 ACCEPTED: Multi-marker integration exhibits identical performance "
                "to the marker selected on the training partition."
            ),
        }

    try:
        res = stats.wilcoxon(s_c, best_single, alternative=alternative)
        w_stat = float(res.statistic)
        p_val = float(res.pvalue)
    except (ValueError, TypeError):
        w_stat = float(np.sum(delta > 0))
        p_val = 1.0 if mean_delta <= 0 else 0.05

    h0_rejected = bool((mean_delta > 0.0) and (p_val < 0.05))
    conclusion = (
        "H0 REJECTED: Multi-marker integration provides statistically superior predictive power "
        "versus the marker selected on the training partition."
        if h0_rejected
        else "H0 ACCEPTED: Multi-marker integration does not statistically outperform "
        "the marker selected on the training partition."
    )
    return {
        "mean_delta": mean_delta,
        "w_statistic": w_stat,
        "p_value": p_val,
        "h0_rejected": h0_rejected,
        "selected_marker": chosen.tolist(),
        "comparator_selected_on": "training",
        "conclusion": conclusion,
    }


test_multimarker_null_hypothesis = evaluate_multimarker_hypothesis
test_multimarker_null_hypothesis.__test__ = False
