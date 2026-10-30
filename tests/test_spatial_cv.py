"""
test_spatial_cv.py - Tests for Milestone 4 spatial cross-validation and leakage guards.
"""

import numpy as np
import pandas as pd
import pytest
from src.spatial_cv.spatial_partitioner import (
    haversine_distance,
    haversine_distance_matrix,
    create_spatial_splits,
    apply_spatial_buffering
)
from src.spatial_cv.leakage_guard import (
    assert_zero_spatial_leakage,
    assert_no_duplicate_site_split,
    verify_all_splits
)
from tests.conftest import oracle_haversine


def test_haversine_matches_oracle():
    lat1, lon1 = 48.8566, 2.3522  # Paris
    lat2, lon2 = 52.5200, 13.4050 # Berlin
    d = haversine_distance(lat1, lon1, lat2, lon2)
    oracle_d = oracle_haversine(lat1, lon1, lat2, lon2)
    assert pytest.approx(d, rel=1e-4) == oracle_d
    assert 850.0 < d < 900.0


def test_haversine_matrix_properties():
    lats = np.array([48.85, 52.52, 41.90])
    lons = np.array([2.35, 13.40, 12.49])
    mat = haversine_distance_matrix(lats, lons)
    assert mat.shape == (3, 3)
    assert np.allclose(np.diag(mat), 0.0)
    assert np.allclose(mat, mat.T)


def test_spatial_group_kfold_zero_leakage():
    # 20 samples with 10 sites (2 samples per site)
    records = []
    for i in range(10):
        records.append({"sample_id": f"S{2*i}", "barcode_id": 2*i, "site_id": f"SITE_{i}"})
        records.append({"sample_id": f"S{2*i+1}", "barcode_id": 2*i+1, "site_id": f"SITE_{i}"})
    df = pd.DataFrame(records)

    splits = create_spatial_splits(df, n_splits=5, group_col="site_id")
    assert len(splits) == 5
    verify_all_splits(df, splits)


def test_leakage_assertion_raises_on_overlap():
    tr_sites = {"SITE_A", "SITE_B", "SITE_C"}
    te_sites = {"SITE_C", "SITE_D"}
    with pytest.raises(AssertionError, match="Spatial data leakage detected"):
        assert_zero_spatial_leakage(tr_sites, te_sites)
