"""
test_models.py - Tests for Milestone 4 classifiers, feature builders, and multi-marker fusion.
"""

import numpy as np
import pytest
from src.models.feature_builder import build_feature_sets, impute_and_scale_inside_fold
from src.models.classifiers import get_classifier, train_classifier, evaluate_predictions
from src.models.multi_marker_fusion import early_fusion, late_fusion_predict


def test_build_feature_sets():
    c16 = np.random.randint(1, 100, size=(10, 20))
    cits = np.random.randint(1, 100, size=(10, 15))
    fsets = build_feature_sets(c16, cits)

    assert "set_a_16s" in fsets
    assert "set_b_its" in fsets
    assert "set_c_integrated" in fsets
    assert fsets["set_a_16s"].shape == (10, 20)
    assert fsets["set_b_its"].shape == (10, 15)
    assert fsets["set_c_integrated"].shape == (10, 35)


def test_impute_and_scale_inside_fold():
    x_tr = np.array([
        [1.0, np.nan],
        [3.0, 10.0],
        [2.0, 20.0]
    ])
    x_te = np.array([
        [np.nan, 15.0]
    ])
    tr_sc, te_sc = impute_and_scale_inside_fold(x_tr, x_te)
    assert not np.isnan(tr_sc).any()
    assert not np.isnan(te_sc).any()


def test_classifiers_and_fusion():
    rng = np.random.RandomState(42)
    x = rng.randn(40, 10)
    y = np.array([0] * 20 + [1] * 20)

    clf = get_classifier("rf", seed=42, n_estimators=20)
    clf = train_classifier(clf, x, y)
    preds = clf.predict(x)
    metrics = evaluate_predictions(y, preds)
    assert metrics["balanced_accuracy"] > 0.8
    assert metrics["macro_f1"] > 0.8

    # Test fusion
    x1 = x[:, :5]
    x2 = x[:, 5:]
    xfused = early_fusion(x1, x2)
    assert xfused.shape == x.shape

    prob1 = np.array([[0.8, 0.2], [0.3, 0.7]])
    prob2 = np.array([[0.6, 0.4], [0.2, 0.8]])
    f_preds, f_probs = late_fusion_predict(prob1, prob2)
    assert np.array_equal(f_preds, [0, 1])
