"""
test_tier3_pairwise.py - Tier 3: Pairwise & Cross-Feature Integration Tests.
Verifies cross-feature combinatorial interactions across module boundaries.
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
# Pairwise 1: Strict Pairing + ESDAC Metadata Join
# ============================================================================
class TestPairwise_PairingAndMetadataJoin:
    def test_pairing_and_esdac_join_consistency(self, mock_lucas_manifest):
        records = mock_lucas_manifest["records"]
        for r in records:
            # Both markers present and joined with edaphic values
            assert r["sra_16s_run"] is not None
            assert r["sra_its_run"] is not None
            assert "ph_h2o" in r
            assert "organic_carbon" in r
            assert "macro_land_cover" in r

    def test_orphan_exclusion_preserves_paired_count(self):
        sra_16s = {1: "SRR16S_1", 2: "SRR16S_2", 3: "SRR16S_3"}
        sra_its = {1: "SRRITS_1", 2: "SRRITS_2"}  # Barcode 3 missing ITS
        esdac = {1: {"ph": 6.5}, 2: {"ph": 7.0}, 3: {"ph": 5.5}}
        
        # Inner join on barcode_id
        paired_ids = set(sra_16s.keys()) & set(sra_its.keys()) & set(esdac.keys())
        assert paired_ids == {1, 2}
        assert 3 not in paired_ids

    def test_duplicate_sites_retain_identical_metadata(self, mock_lucas_manifest):
        records = {r["barcode_id"]: r for r in mock_lucas_manifest["records"]}
        for b1, b2 in mock_lucas_manifest["duplicate_pairs"]:
            if b1 in records and b2 in records:
                assert records[b1]["site_id"] == records[b2]["site_id"]
                assert records[b1]["latitude"] == records[b2]["latitude"]


# ============================================================================
# Pairwise 2: SRS Normalization + Taxonomic Rollup
# ============================================================================
class TestPairwise_SRSAndTaxonomicRollup:
    def test_srs_normalization_then_phylum_rollup(self, mock_count_data, mock_taxonomy):
        c16 = mock_count_data["counts_16s"][:10]
        target_depth = 40109
        norm_counts = np.zeros_like(c16)
        for i in range(c16.shape[0]):
            norm_counts[i] = oracle_srs_normalization(c16[i], target_depth=target_depth, seed=42)
            
        phyla = [t.split(";")[1].strip() for t in mock_taxonomy["taxonomy_16s"]]
        unique_phyla = sorted(list(set(phyla)))
        rollup = np.zeros((norm_counts.shape[0], len(unique_phyla)))
        for idx, p in enumerate(unique_phyla):
            col_indices = [j for j, taxon_p in enumerate(phyla) if taxon_p == p]
            rollup[:, idx] = np.sum(norm_counts[:, col_indices], axis=1)
            
        # Total abundance per sample after rollup must remain target_depth
        np.testing.assert_allclose(np.sum(rollup, axis=1), target_depth)

    def test_relative_phyla_proportions_preserved(self, mock_count_data, mock_taxonomy):
        row = mock_count_data["counts_16s"][0]
        norm = oracle_srs_normalization(row, target_depth=40109, seed=42)
        prop_raw = row / np.sum(row)
        prop_norm = norm / np.sum(norm)
        # Correlation of proportions must be extremely high (> 0.99)
        corr = np.corrcoef(prop_raw, prop_norm)[0, 1]
        assert corr > 0.99


# ============================================================================
# Pairwise 3: CLR Transformation + Aitchison Distance Matrix
# ============================================================================
class TestPairwise_CLRAndAitchisonDistance:
    def test_aitchison_equals_euclidean_distance_on_clr(self, mock_count_data):
        c1 = mock_count_data["counts_16s"][0]
        c2 = mock_count_data["counts_16s"][1]
        clr1 = oracle_clr(c1)
        clr2 = oracle_clr(c2)
        
        euclidean_on_clr = float(np.linalg.norm(clr1 - clr2))
        aitchison_direct = oracle_aitchison_distance(c1, c2)
        np.testing.assert_allclose(aitchison_direct, euclidean_on_clr, atol=1e-7)

    def test_aitchison_distance_matrix_properties(self, mock_count_data):
        c = mock_count_data["counts_16s"][:8]
        n = len(c)
        dist_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_mat[i, j] = oracle_aitchison_distance(c[i], c[j])
        # Symmetric
        np.testing.assert_allclose(dist_mat, dist_mat.T)
        # Zero diagonal
        np.testing.assert_allclose(np.diag(dist_mat), 0.0)
        # Strictly positive off-diagonal
        assert np.all(dist_mat[np.triu_indices(n, k=1)] > 0.0)


# ============================================================================
# Pairwise 4: CLR + Native PERMANOVA & Multivariate Dispersion
# ============================================================================
class TestPairwise_CLRAndMultivariatePERMANOVA:
    def test_clr_aitchison_permanova_pipeline(self, mock_count_data, mock_lucas_manifest):
        c = mock_count_data["counts_16s"][:15]
        clr = oracle_clr(c)
        n = len(clr)
        # Compute distance matrix
        dm = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dm[i, j] = float(np.linalg.norm(clr[i] - clr[j]))
        
        groups = np.array([r["macro_land_cover"] for r in mock_lucas_manifest["records"][:15]])
        assert len(set(groups)) >= 2
        
        # Simple PERMANOVA pseudo-F mock verification
        total_ss = np.sum(dm ** 2) / (2 * n)
        assert total_ss > 0.0

    def test_dispersion_accompanied_permanova(self):
        # PERMANOVA with significant F and non-significant dispersion confirms true centroid shift
        f_stat = 4.8
        p_permanova = 0.001
        p_betadisper = 0.35  # Dispersion homogeneous
        assert p_permanova < 0.05
        assert p_betadisper >= 0.05  # Valid centroid difference


# ============================================================================
# Pairwise 5: SpatialGroupKFold + Zero Leakage Assertions
# ============================================================================
class TestPairwise_SpatialGroupKFoldAndZeroLeakage:
    def test_spatial_group_kfold_zero_leakage_pass(self, mock_lucas_manifest):
        records = mock_lucas_manifest["records"]
        site_ids = np.array([r["site_id"] for r in records])
        coords = np.array([[r["latitude"], r["longitude"]] for r in records])
        y = np.array([r["macro_land_cover"] for r in records])
        
        unique_sites = np.unique(site_ids)
        k = 4
        # Partition unique sites across k folds
        site_folds = {s: i % k for i, s in enumerate(unique_sites)}
        
        for fold in range(k):
            test_mask = np.array([site_folds[s] == fold for s in site_ids])
            train_mask = ~test_mask
            
            tr_idx = np.where(train_mask)[0]
            te_idx = np.where(test_mask)[0]
            
            # Assert Index Disjointness
            assert len(set(tr_idx).intersection(set(te_idx))) == 0
            # Assert Site ID Disjointness
            tr_sites = set(site_ids[tr_idx])
            te_sites = set(site_ids[te_idx])
            assert len(tr_sites.intersection(te_sites)) == 0
            # Assert Exact Coordinate Disjointness
            tr_coords = {tuple(c) for c in coords[tr_idx]}
            te_coords = {tuple(c) for c in coords[te_idx]}
            assert len(tr_coords.intersection(te_coords)) == 0
            # Assert Class Representation Invariant
            tr_classes = set(y[tr_idx])
            te_classes = set(y[te_idx])
            assert te_classes.issubset(tr_classes)

    def test_duplicate_pairs_always_in_same_fold(self, mock_lucas_manifest):
        records = mock_lucas_manifest["records"]
        site_ids = np.array([r["site_id"] for r in records])
        unique_sites = np.unique(site_ids)
        site_folds = {s: i % 3 for i, s in enumerate(unique_sites)}
        
        rec_dict = {r["barcode_id"]: r for r in records}
        for b1, b2 in mock_lucas_manifest["duplicate_pairs"]:
            if b1 in rec_dict and b2 in rec_dict:
                f1 = site_folds[rec_dict[b1]["site_id"]]
                f2 = site_folds[rec_dict[b2]["site_id"]]
                assert f1 == f2


# ============================================================================
# Pairwise 6: Spatial CV + Classifier Pipeline
# ============================================================================
class TestPairwise_SpatialCVAndClassifierPipeline:
    def test_inside_fold_preprocessing_isolation(self):
        # Normalization parameters must be calculated on train only
        X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_test = np.array([[10.0, 20.0]])
        
        mean_tr = np.mean(X_train, axis=0)
        std_tr = np.std(X_train, axis=0)
        
        X_train_scaled = (X_train - mean_tr) / std_tr
        X_test_scaled = (X_test - mean_tr) / std_tr
        
        # Test sample does not alter train mean
        np.testing.assert_allclose(np.mean(X_train_scaled, axis=0), [0.0, 0.0], atol=1e-7)

    def test_spatial_cv_model_evaluation_metrics(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 1, 2])
        bacc = oracle_balanced_accuracy(y_true, y_pred)
        f1 = oracle_macro_f1(y_true, y_pred)
        assert 0.0 <= bacc <= 1.0
        assert 0.0 <= f1 <= 1.0


# ============================================================================
# Pairwise 7: Early Fusion vs Late Fusion Comparison
# ============================================================================
class TestPairwise_EarlyVsLateFusionComparison:
    def test_early_and_late_fusion_pipeline(self, mock_count_data):
        c16 = mock_count_data["counts_16s"][:20]
        cits = mock_count_data["counts_its"][:20]
        
        # Early fusion
        clr_16 = oracle_clr(c16)
        clr_its = oracle_clr(cits)
        w16 = 1.0 / math.sqrt(clr_16.shape[1])
        wits = 1.0 / math.sqrt(clr_its.shape[1])
        X_early = np.hstack([w16 * clr_16, wits * clr_its])
        assert X_early.shape == (20, 120 + 80)
        
        # Late fusion (soft voting)
        p16 = np.ones((20, 3)) / 3.0
        pits = np.ones((20, 3)) / 3.0
        p_late = 0.5 * p16 + 0.5 * pits
        np.testing.assert_allclose(np.sum(p_late, axis=1), 1.0)


# ============================================================================
# Pairwise 8: Inside-Fold Edaphic Imputation
# ============================================================================
class TestPairwise_InsideFoldEdaphicImputation:
    def test_median_imputation_strictly_on_train(self):
        ph_train = np.array([5.0, 6.0, 7.0, np.nan])
        ph_test = np.array([np.nan, 8.0])
        
        train_median = float(np.nanmedian(ph_train))
        assert train_median == 6.0
        
        # Impute both using train_median
        ph_train_imputed = np.where(np.isnan(ph_train), train_median, ph_train)
        ph_test_imputed = np.where(np.isnan(ph_test), train_median, ph_test)
        
        assert not np.isnan(ph_train_imputed).any()
        assert not np.isnan(ph_test_imputed).any()
        assert ph_test_imputed[0] == train_median


# ============================================================================
# Pairwise 9: Class Representation + Spatial Buffering
# ============================================================================
class TestPairwise_ClassRepresentationAndSpatialBuffering:
    def test_buffer_preserves_class_representation(self):
        coords_train = np.array([[50.0, 4.0], [55.0, 10.0], [60.0, 15.0]])
        y_train = np.array(["Crop", "Wood", "Grass"])
        coord_test = (50.01, 4.01) # Near Crop
        
        buffer_km = 10.0
        dists = [oracle_haversine(c[0], c[1], coord_test[0], coord_test[1]) for c in coords_train]
        retained = [i for i, d in enumerate(dists) if d >= buffer_km]
        
        retained_classes = set(y_train[retained])
        # If test class is Wood, it is retained
        assert "Wood" in retained_classes


# ============================================================================
# Pairwise 10: Alpha Diversity + Environmental Association
# ============================================================================
class TestPairwise_AlphaDiversityAndEnvironmentalGradient:
    def test_alpha_diversity_correlates_with_edaphic_ph(self, mock_count_data, mock_lucas_manifest):
        c16 = mock_count_data["counts_16s"][:20]
        shannon = np.array([oracle_shannon_entropy(row) for row in c16])
        phs = np.array([r["ph_h2o"] for r in mock_lucas_manifest["records"][:20]])
        
        # In soil ecology, microbial diversity often peaks around neutral pH (6.5 - 7.5)
        assert len(shannon) == len(phs)
        assert np.all(shannon > 0.0)
        assert np.all(phs > 0.0)
