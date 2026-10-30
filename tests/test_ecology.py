"""
test_ecology.py - Tests for Milestone 3 compositional ecology and multivariate stats modules.
"""

import numpy as np
import pytest
from src.ecology.alpha_diversity import (
    calculate_observed_richness,
    calculate_shannon,
    calculate_simpson,
    calculate_chao1,
    compute_alpha_diversity_table
)
from src.ecology.compositional import clr_transform, robust_clr_transform
from src.ecology.beta_diversity import bray_curtis_distance_matrix, aitchison_distance_matrix
from src.ecology.multivariate_stats import run_permanova, run_betadisper
from tests.conftest import (
    oracle_clr,
    oracle_shannon_entropy,
    oracle_simpson_index,
    oracle_chao1,
    oracle_aitchison_distance,
    oracle_bray_curtis
)


def test_alpha_diversity_indices():
    counts = np.array([10, 5, 2, 1, 0, 0, 1])
    assert calculate_observed_richness(counts) == 5
    assert pytest.approx(calculate_shannon(counts), 1e-5) == oracle_shannon_entropy(counts)
    assert pytest.approx(calculate_simpson(counts), 1e-5) == oracle_simpson_index(counts)
    assert pytest.approx(calculate_chao1(counts), 1e-5) == oracle_chao1(counts)


def test_clr_transformation_exact_oracle():
    rng = np.random.RandomState(42)
    mat = rng.randint(0, 100, size=(10, 20))
    clr_res = clr_transform(mat, pseudocount=1e-5)
    oracle_res = oracle_clr(mat, pseudocount=1e-5)

    assert np.allclose(clr_res, oracle_res)
    # Sum across features should be zero for every sample
    assert np.allclose(np.sum(clr_res, axis=1), 0.0)


def test_beta_diversity_matrices():
    mat = np.array([
        [10, 20, 30],
        [15, 25, 35],
        [0, 5, 50]
    ])
    bc = bray_curtis_distance_matrix(mat)
    assert bc.shape == (3, 3)
    assert np.allclose(np.diag(bc), 0.0)
    assert np.allclose(bc, bc.T)
    assert pytest.approx(bc[0, 1], 1e-5) == oracle_bray_curtis(mat[0], mat[1])

    ait = aitchison_distance_matrix(mat, pseudocount=1e-3)
    assert ait.shape == (3, 3)
    assert np.allclose(np.diag(ait), 0.0)
    assert np.allclose(ait, ait.T)
    assert pytest.approx(ait[0, 1], 1e-5) == oracle_aitchison_distance(mat[0], mat[1], pseudocount=1e-3)


def test_permanova_and_betadisper():
    rng = np.random.RandomState(42)
    groupings = np.array([0] * 8 + [1] * 8)
    # Create synthetic clustered points
    pts = np.vstack([
        rng.normal(loc=0.0, scale=0.5, size=(8, 5)),
        rng.normal(loc=5.0, scale=0.5, size=(8, 5))
    ])
    dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)

    perm_res = run_permanova(dists, groupings, permutations=199, seed=42)
    assert "f_statistic" in perm_res
    assert "p_value" in perm_res
    assert perm_res["f_statistic"] > 5.0  # Clear cluster separation
    assert perm_res["p_value"] < 0.05

    disp_res = run_betadisper(dists, groupings)
    assert "homogeneous" in disp_res
