"""
test_evaluation.py - Tests for Milestone 4 evaluation metrics, bootstrap CIs, and H0 hypothesis testing.
"""

import numpy as np
import pytest
from src.evaluation.metrics import compute_classification_metrics, compute_confusion_matrix_summary
from src.evaluation.bootstrap_ci import compute_spatial_bootstrap_ci
from src.evaluation.hypothesis_test import evaluate_multimarker_hypothesis
from tests.conftest import oracle_balanced_accuracy, oracle_macro_f1


def test_metrics_match_oracle():
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 0, 1])

    metrics = compute_classification_metrics(y_true, y_pred)
    assert pytest.approx(metrics["balanced_accuracy"], 1e-4) == oracle_balanced_accuracy(y_true, y_pred)
    assert pytest.approx(metrics["macro_f1"], 1e-4) == oracle_macro_f1(y_true, y_pred)


def test_bootstrap_ci():
    rng = np.random.RandomState(42)
    scores = rng.normal(loc=0.75, scale=0.03, size=20)
    ci = compute_spatial_bootstrap_ci(scores, n_bootstraps=500, seed=42)

    assert ci["ci_lower"] < ci["mean"] < ci["ci_upper"]
    assert 0.70 < ci["ci_lower"] < 0.80


def test_null_hypothesis_superiority():
    # Set C strictly higher
    s_a = np.array([0.70, 0.72, 0.71, 0.73, 0.70, 0.72])
    s_b = np.array([0.65, 0.67, 0.66, 0.68, 0.65, 0.66])
    s_c = np.array([0.80, 0.82, 0.81, 0.83, 0.80, 0.81])

    res = evaluate_multimarker_hypothesis(s_a, s_b, s_c, alternative="greater")
    assert res["mean_delta"] > 0
    assert res["p_value"] < 0.05
    assert res["h0_rejected"] is True

    # Set C equal/lower
    s_c_low = np.array([0.69, 0.71, 0.70, 0.72, 0.69, 0.71])
    res_low = evaluate_multimarker_hypothesis(s_a, s_b, s_c_low, alternative="greater")
    assert res_low["h0_rejected"] is False
