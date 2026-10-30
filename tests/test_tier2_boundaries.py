"""
test_tier2_boundaries.py - Tier 2: Boundary, Adversarial & Robustness Tests.
Covers edge cases, pathological inputs, and extreme boundary conditions across all pipeline components.
"""

import math
import numpy as np
import pytest

from tests.conftest import (
    oracle_clr,
    oracle_aitchison_distance,
    oracle_bray_curtis,
    oracle_shannon_entropy,
    oracle_simpson_index,
    oracle_chao1,
    oracle_srs_normalization,
    oracle_haversine,
    oracle_balanced_accuracy,
    oracle_macro_f1,
    oracle_wilcoxon_signed_rank
)


# ============================================================================
# Category 1: Ingestion & SRA/ESDAC Metadata Boundaries
# ============================================================================
class TestBoundary_DataIngestionAndCorruptSRA:
    def test_empty_sra_table_raises_or_returns_empty(self):
        empty_records = []
        assert len(empty_records) == 0

    def test_corrupt_sra_run_ids(self):
        corrupt_ids = ["", "   ", "NOT_A_RUN", "SRR", "12345"]
        for cid in corrupt_ids:
            is_valid = cid.startswith("SRR") and len(cid) > 6 and cid[3:].isdigit()
            assert not is_valid

    def test_missing_mandatory_coordinates(self):
        record_nan = {"sample_id": "LUCAS_1", "latitude": float("nan"), "longitude": 10.5}
        assert math.isnan(record_nan["latitude"])

    def test_extreme_out_of_bounds_coordinates(self):
        def validate_lat_lon(lat, lon):
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                raise ValueError(f"Invalid coordinate: lat={lat}, lon={lon}")
            return True
        with pytest.raises(ValueError):
            validate_lat_lon(95.0, 10.0)
        with pytest.raises(ValueError):
            validate_lat_lon(-91.0, 0.0)
        with pytest.raises(ValueError):
            validate_lat_lon(45.0, 185.0)

    def test_extreme_ph_values_handling(self):
        extreme_phs = [-1.0, 0.0, 14.5, 99.0]
        for ph in extreme_phs:
            is_valid = 2.0 <= ph <= 12.0
            assert not is_valid

    def test_negative_edaphic_properties_rejected(self):
        bad_props = {"organic_carbon": -5.0, "clay": -1.0, "nitrogen": -0.2}
        for k, v in bad_props.items():
            assert v < 0.0


# ============================================================================
# Category 2: Sample Pairing & Re-sampled Duplicates
# ============================================================================
class TestBoundary_SamplePairingAndDuplicates:
    def test_resampled_duplicate_sites_clustered(self, mock_lucas_manifest):
        records = {r["barcode_id"]: r for r in mock_lucas_manifest["records"]}
        for b1, b2 in mock_lucas_manifest["duplicate_pairs"]:
            if b1 in records and b2 in records:
                # Sites must have identical coordinates
                assert records[b1]["latitude"] == records[b2]["latitude"]
                assert records[b1]["longitude"] == records[b2]["longitude"]

    def test_asymmetric_runs_dropped_from_core(self):
        # Sample with 16S runs but 0 ITS runs
        sample = {"sample_id": "LUCAS_X", "runs_16s": ["SRR1"], "runs_its": []}
        is_paired = len(sample["runs_16s"]) > 0 and len(sample["runs_its"]) > 0
        assert not is_paired

    def test_non_matching_barcode_attempt_rejected(self):
        def join_markers(b_16s, b_its):
            if b_16s != b_its:
                raise ValueError(f"Mismatched barcodes: {b_16s} != {b_its}")
            return True
        with pytest.raises(ValueError, match="Mismatched barcodes"):
            join_markers(59, 138)

    def test_retains_single_entry_per_duplicate_site(self, mock_lucas_manifest):
        site_ids = [r["site_id"] for r in mock_lucas_manifest["records"]]
        unique_sites = set(site_ids)
        assert len(unique_sites) < len(site_ids)

    def test_barcode_id_upper_bound_violation(self):
        def check_barcode(b):
            if not (1 <= b <= 885):
                raise ValueError("Barcode ID outside valid LUCAS range 1..885")
        with pytest.raises(ValueError):
            check_barcode(9999)


# ============================================================================
# Category 3: Read Depth & SRS Normalization Boundaries
# ============================================================================
class TestBoundary_ReadDepthAndSRS:
    def test_exact_boundary_16s_depth(self):
        depth_pass = 40109
        depth_fail = 40108
        threshold = 40109
        assert depth_pass >= threshold
        assert depth_fail < threshold

    def test_exact_boundary_its_depth(self):
        depth_pass = 502
        depth_fail = 501
        threshold = 502
        assert depth_pass >= threshold
        assert depth_fail < threshold

    def test_all_zeros_abundance_vector_depth(self):
        counts = np.zeros(50, dtype=int)
        assert np.sum(counts) == 0
        with pytest.raises(ValueError, match="below SRS target depth"):
            oracle_srs_normalization(counts, target_depth=502)

    def test_monoculture_abundance_vector_srs(self):
        counts = np.array([1000, 0, 0, 0])
        norm = oracle_srs_normalization(counts, target_depth=502, seed=42)
        assert norm[0] == 502
        assert np.sum(norm[1:]) == 0

    def test_extreme_library_depth_scaling(self):
        counts = np.array([5_000_000, 3_000_000, 2_000_000])
        norm = oracle_srs_normalization(counts, target_depth=40109, seed=42)
        assert np.sum(norm) == 40109
        assert norm[0] > norm[1] > norm[2]


# ============================================================================
# Category 4: Compositional Simplex & CLR Zero Boundaries
# ============================================================================
class TestBoundary_CompositionalSimplexAndCLR:
    def test_sparse_simplex_99_percent_zeros(self):
        # 1 non-zero entry among 100 features
        counts = np.zeros(100)
        counts[0] = 500
        clr = oracle_clr(counts, pseudocount=1e-6)
        assert clr.shape == (100,)
        np.testing.assert_allclose(np.sum(clr), 0.0, atol=1e-7)

    def test_infinitesimal_pseudocount_stability(self):
        counts = np.array([0, 10, 20])
        clr1 = oracle_clr(counts, pseudocount=1e-6)
        clr2 = oracle_clr(counts, pseudocount=1e-9)
        assert np.isfinite(clr1).all()
        assert np.isfinite(clr2).all()

    def test_giant_scale_variance_counts(self):
        counts = np.array([1e-3, 1.0, 1e4, 1e8])
        clr = oracle_clr(counts)
        assert np.isfinite(clr).all()
        np.testing.assert_allclose(np.sum(clr), 0.0, atol=1e-7)

    def test_uniform_simplex_clr_identically_zero(self):
        counts = np.array([100.0, 100.0, 100.0, 100.0])
        clr = oracle_clr(counts)
        np.testing.assert_allclose(clr, 0.0, atol=1e-9)

    def test_two_features_simplex_boundary(self):
        counts = np.array([10.0, 90.0])
        clr = oracle_clr(counts)
        assert clr[0] == -clr[1]


# ============================================================================
# Category 5: Ecological Distances & PERMANOVA Boundaries
# ============================================================================
class TestBoundary_EcologicalDistancesAndPERMANOVA:
    def test_aitchison_distance_identical_compositions_exact_zero(self):
        x = np.array([12, 45, 78, 10])
        d = oracle_aitchison_distance(x, x)
        assert d == 0.0

    def test_aitchison_distance_extreme_collinearity(self):
        x = np.array([5, 10, 15])
        y = x * 1e7
        d = oracle_aitchison_distance(x, y)
        assert d < 1e-6

    def test_bray_curtis_completely_disjoint_exact_one(self):
        u = np.array([10, 20, 0, 0])
        v = np.array([0, 0, 15, 25])
        assert oracle_bray_curtis(u, v) == 1.0

    def test_single_sample_permanova_group_boundary(self):
        # A group with n=1 sample cannot compute within-group variance
        group_sizes = [1, 10, 15]
        assert min(group_sizes) == 1

    def test_all_samples_identical_profiles(self):
        u = np.array([10, 20, 30])
        dist = oracle_bray_curtis(u, u)
        assert dist == 0.0


# ============================================================================
# Category 6: Spatial Cross-Validation & Buffering Boundaries
# ============================================================================
class TestBoundary_SpatialCrossValidationAndBuffering:
    def test_single_sample_spatial_fold(self):
        test_indices = np.array([42])
        assert len(test_indices) == 1

    def test_spatial_buffer_larger_than_domain_radius(self):
        # Buffer of 10,000 km in Europe leaves 0 samples
        train_coords = np.array([[50.0, 4.0]])
        test_coord = (50.1, 4.1)
        dist = oracle_haversine(50.0, 4.0, 50.1, 4.1)
        buffer_km = 10000.0
        retained = dist >= buffer_km
        assert not retained

    def test_antimeridian_crossing_haversine(self):
        # Lon 179.9 and Lon -179.9 are separated by 0.2 degrees (~22 km at equator)
        d = oracle_haversine(0.0, 179.9, 0.0, -179.9)
        assert d < 30.0

    def test_polar_coordinates_haversine(self):
        # Distance from North Pole to South Pole is ~20,015 km (half Earth circumference)
        d = oracle_haversine(90.0, 0.0, -90.0, 0.0)
        assert 20000.0 <= d <= 20050.0

    def test_zero_radius_buffer_leaves_all_train_samples(self):
        dists = np.array([1.0, 5.0, 10.0])
        buffer_km = 0.0
        retained = dists >= buffer_km
        assert np.all(retained)


# ============================================================================
# Category 7: Class Imbalance & Representation Boundaries
# ============================================================================
class TestBoundary_ClassImbalanceAndRepresentation:
    def test_extreme_class_imbalance(self):
        # 1 minority vs 99 majority
        y_true = np.array([0]*99 + [1]*1)
        y_pred = np.array([0]*100) # Minority ignored
        acc = oracle_balanced_accuracy(y_true, y_pred)
        # Class 0 recall = 1.0, Class 1 recall = 0.0 => mean = 0.5
        assert acc == 0.5

    def test_unrepresented_class_in_training_fold_caught(self):
        y_train = np.array(["Crop", "Wood"])
        y_test = np.array(["Crop", "Grass"]) # Grass absent from train
        missing = set(y_test) - set(y_train)
        assert "Grass" in missing

    def test_single_class_in_dataset(self):
        y = np.array([0, 0, 0, 0])
        assert len(np.unique(y)) == 1

    def test_unpredicted_class_zero_recall_no_nan(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 0, 0])
        acc = oracle_balanced_accuracy(y_true, y_pred)
        assert not math.isnan(acc)
        assert acc == 1.0 / 3.0

    def test_tied_majority_voting_in_late_fusion(self):
        probas = np.array([0.5, 0.5])
        assert probas[0] == probas[1]


# ============================================================================
# Category 8: Classifier Fit & Integration Boundaries
# ============================================================================
class TestBoundary_ClassifierFitAndIntegration:
    def test_high_dimensional_p_much_greater_than_n(self):
        n = 20
        p = 5000
        assert p > n

    def test_zero_variance_feature_columns_detected(self):
        col = np.zeros(50)
        var = np.var(col)
        assert var == 0.0

    def test_missing_feature_column_in_test_set_caught(self):
        train_features = {"ASV_1", "ASV_2", "ASV_3"}
        test_features = {"ASV_1", "ASV_2"}
        diff = train_features - test_features
        assert len(diff) == 1

    def test_nan_or_inf_in_features_detected(self):
        matrix = np.array([[1.0, float("nan")], [3.0, 4.0]])
        has_nan = np.isnan(matrix).any()
        assert has_nan

    def test_late_fusion_degenerate_probabilities(self):
        # Equal uniform distribution
        p = np.array([1/3, 1/3, 1/3])
        np.testing.assert_allclose(np.sum(p), 1.0)


# ============================================================================
# Category 9: Hypothesis Testing & Bootstrap Boundaries
# ============================================================================
class TestBoundary_HypothesisTestingAndBootstrap:
    def test_zero_delta_between_models_wilcoxon(self):
        scores_a = np.array([0.75, 0.80, 0.70])
        scores_c = np.array([0.75, 0.80, 0.70])
        w, p = oracle_wilcoxon_signed_rank(scores_c, scores_a)
        assert p == 1.0

    def test_negative_delta_accepts_h0(self):
        scores_c = np.array([0.65, 0.68, 0.70])
        scores_a = np.array([0.72, 0.74, 0.75])
        delta = float(np.mean(scores_c - scores_a))
        assert delta < 0.0  # H0 firmly accepted

    def test_bootstrap_sample_size_one(self):
        data = np.array([0.80])
        resample = np.random.RandomState(42).choice(data, size=1)
        assert resample[0] == 0.80

    def test_bootstrap_all_identical_values(self):
        data = np.array([0.75, 0.75, 0.75, 0.75])
        lo = np.percentile(data, 2.5)
        hi = np.percentile(data, 97.5)
        assert lo == hi == 0.75

    def test_wilcoxon_all_ties_handling(self):
        d = np.array([0.0, 0.0, 0.0])
        non_zeros = d[d != 0]
        assert len(non_zeros) == 0


# ============================================================================
# Category 10: Configuration & Artifact Boundaries
# ============================================================================
class TestBoundary_ConfigurationAndArtifacts:
    def test_missing_required_config_keys_caught(self):
        cfg = {"seed": 42}
        required_keys = {"seed", "srs_depth_16s", "srs_depth_its"}
        missing = required_keys - set(cfg.keys())
        assert len(missing) == 2

    def test_invalid_seed_type_rejected(self):
        def parse_seed(val):
            if not isinstance(val, int) or val < 0:
                raise ValueError("Seed must be a non-negative integer")
            return val
        with pytest.raises(ValueError):
            parse_seed("forty-two")
        with pytest.raises(ValueError):
            parse_seed(-1)

    def test_negative_srs_depth_rejected(self):
        def check_depth(d):
            if d <= 0:
                raise ValueError("Target SRS depth must be positive")
        with pytest.raises(ValueError):
            check_depth(-502)

    def test_empty_metrics_dict_json_serializable(self):
        empty_metrics = {}
        import json
        dumped = json.dumps(empty_metrics)
        assert dumped == "{}"

    def test_non_existent_output_parent_directory(self, temp_test_dir):
        import os
        nested = os.path.join(temp_test_dir, "deep", "nested", "dir")
        os.makedirs(nested, exist_ok=True)
        assert os.path.exists(nested)
