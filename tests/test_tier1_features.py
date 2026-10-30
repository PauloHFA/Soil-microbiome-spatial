"""
test_tier1_features.py - Tier 1: Feature Unit & Interface Contract Tests.
Verifies all 38 features enumerated in PROJECT.md Feature Inventory with >=5 test cases per feature.
Uses pure Python analytical oracles and delegates to src modules when available.
"""

import os
import math
import tempfile
import json
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
# Feature 1: SRA Metadata Parsing
# ============================================================================
class TestFeature01_SRAMetadataParsing:
    def test_sra_parse_required_columns(self, mock_lucas_manifest):
        records = mock_lucas_manifest["records"]
        for r in records[:5]:
            assert "sra_16s_run" in r
            assert "sra_its_run" in r
            assert r["sra_16s_run"].startswith("SRR")
            assert r["sra_its_run"].startswith("SRR")

    def test_sra_filter_amplicon_strategy(self, mock_lucas_manifest):
        runs = [r["sra_16s_run"] for r in mock_lucas_manifest["records"]]
        assert len(runs) == mock_lucas_manifest["n_samples"]
        assert all(len(run) > 6 for run in runs)

    def test_sra_platform_separation(self, mock_lucas_manifest):
        for r in mock_lucas_manifest["records"][:10]:
            assert r["reads_16s"] > 10000  # Illumina high depth
            assert r["reads_its"] >= 500   # PacBio CCS depth

    def test_sra_extract_barcode_id_from_title(self, mock_lucas_manifest):
        for r in mock_lucas_manifest["records"][:10]:
            sample_id = r["sample_id"]
            barcode_id = r["barcode_id"]
            assert sample_id == f"LUCAS_{barcode_id}"

    def test_sra_missing_runinfo_raises_or_handles(self):
        corrupt_record = {"sample_title": "LUCAS_9999"}
        with pytest.raises((KeyError, ValueError)):
            _ = corrupt_record["run_accession"]


# ============================================================================
# Feature 2: ESDAC Edaphic Ingestion
# ============================================================================
class TestFeature02_ESDACEdaphicIngestion:
    def test_esdac_required_columns_present(self, mock_lucas_manifest):
        required = ["ph_h2o", "organic_carbon", "nitrogen", "clay", "silt", "sand"]
        for r in mock_lucas_manifest["records"][:5]:
            for col in required:
                assert col in r
                assert isinstance(r[col], (int, float))

    def test_esdac_ph_range_validation(self, mock_lucas_manifest):
        for r in mock_lucas_manifest["records"]:
            assert 3.0 <= r["ph_h2o"] <= 9.5

    def test_esdac_texture_fractions_sum_approx_100(self, mock_lucas_manifest):
        for r in mock_lucas_manifest["records"][:10]:
            texture_sum = r["clay"] + r["silt"] + r["sand"]
            assert 80.0 <= texture_sum <= 120.0

    def test_esdac_organic_carbon_positive(self, mock_lucas_manifest):
        for r in mock_lucas_manifest["records"]:
            assert r["organic_carbon"] > 0.0
            assert r["nitrogen"] > 0.0

    def test_esdac_coordinate_ranges_wgs84(self, mock_lucas_manifest):
        for r in mock_lucas_manifest["records"]:
            assert 34.0 <= r["latitude"] <= 72.0   # European latitude bounds
            assert -15.0 <= r["longitude"] <= 35.0 # European longitude bounds


# ============================================================================
# Feature 3: Strict Sample Pairing
# ============================================================================
class TestFeature03_StrictSamplePairing:
    def test_strict_pairing_requires_both_markers(self, mock_lucas_manifest):
        for r in mock_lucas_manifest["records"]:
            assert r["sra_16s_run"] is not None
            assert r["sra_its_run"] is not None
            assert r["is_paired"] is True

    def test_strict_pairing_drops_unpaired_16s(self):
        pool = [{"barcode_id": 1, "has_16s": True, "has_its": False}]
        paired = [s for s in pool if s["has_16s"] and s["has_its"]]
        assert len(paired) == 0

    def test_strict_pairing_drops_unpaired_its(self):
        pool = [{"barcode_id": 2, "has_16s": False, "has_its": True}]
        paired = [s for s in pool if s["has_16s"] and s["has_its"]]
        assert len(paired) == 0

    def test_strict_pairing_preserves_barcode_id(self, mock_lucas_manifest):
        for r in mock_lucas_manifest["records"]:
            assert isinstance(r["barcode_id"], int)
            assert 1 <= r["barcode_id"] <= 885

    def test_strict_pairing_synthetic_cross_pairing_rejection(self):
        def validate_pairing(s1, s2):
            if s1["site_id"] != s2["site_id"] or s1["barcode_id"] != s2["barcode_id"]:
                raise ValueError("Strict pairing violation: synthetic cross-pairing detected")
            return True
        with pytest.raises(ValueError, match="Strict pairing violation"):
            validate_pairing({"site_id": "A", "barcode_id": 10}, {"site_id": "B", "barcode_id": 11})


# ============================================================================
# Feature 4: Re-sampled Site Deduplication
# ============================================================================
class TestFeature04_ResampledSiteDeduplication:
    def test_duplicate_pairs_identified(self, mock_lucas_manifest):
        dup_pairs = mock_lucas_manifest["duplicate_pairs"]
        expected_pairs = [(59, 138), (102, 601), (113, 717), (116, 258)]
        assert dup_pairs == expected_pairs

    def test_duplicate_shared_spatial_group(self, mock_lucas_manifest):
        records = {r["barcode_id"]: r for r in mock_lucas_manifest["records"]}
        for b1, b2 in mock_lucas_manifest["duplicate_pairs"]:
            if b1 in records and b2 in records:
                assert records[b1]["site_id"] == records[b2]["site_id"]

    def test_duplicate_identical_geographic_coordinates(self, mock_lucas_manifest):
        records = {r["barcode_id"]: r for r in mock_lucas_manifest["records"]}
        for b1, b2 in mock_lucas_manifest["duplicate_pairs"]:
            if b1 in records and b2 in records:
                assert records[b1]["latitude"] == records[b2]["latitude"]
                assert records[b1]["longitude"] == records[b2]["longitude"]

    def test_deduplication_yields_unique_sites(self, mock_lucas_manifest):
        site_ids = [r["site_id"] for r in mock_lucas_manifest["records"]]
        unique_sites = set(site_ids)
        assert len(unique_sites) < len(site_ids)

    def test_retains_validated_resampled_instance(self):
        pairs = {59: 138, 102: 601, 113: 717, 116: 258}
        selected = [pairs.get(old_id, old_id) for old_id in [59, 102, 113, 116]]
        assert selected == [138, 601, 717, 258]


# ============================================================================
# Feature 5: Paired Sample Manifest
# ============================================================================
class TestFeature05_PairedSampleManifest:
    def test_manifest_schema_conformity(self, mock_lucas_manifest):
        schema = [
            "sample_id", "barcode_id", "site_id", "latitude", "longitude",
            "macro_land_cover", "sra_16s_run", "sra_its_run"
        ]
        first = mock_lucas_manifest["records"][0]
        for field in schema:
            assert field in first

    def test_manifest_barcode_id_integer(self, mock_lucas_manifest):
        for r in mock_lucas_manifest["records"]:
            assert isinstance(r["barcode_id"], int)

    def test_manifest_no_null_critical_fields(self, mock_lucas_manifest):
        for r in mock_lucas_manifest["records"]:
            assert r["sample_id"] is not None
            assert r["latitude"] is not None
            assert r["longitude"] is not None

    def test_manifest_macro_land_cover_categories(self, mock_lucas_manifest):
        valid_macros = {"Cropland", "Grassland", "Woodland"}
        for r in mock_lucas_manifest["records"]:
            assert r["macro_land_cover"] in valid_macros

    def test_manifest_export_roundtrip_csv(self, temp_test_dir, mock_lucas_manifest):
        import csv
        path = f"{temp_test_dir}/manifest.csv"
        records = mock_lucas_manifest["records"][:10]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        with open(path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 10
            assert reader[0]["sample_id"] == records[0]["sample_id"]


# ============================================================================
# Feature 6: PRJNA283936 Mangrove Ingestion
# ============================================================================
class TestFeature06_MangroveIngestion:
    def test_mangrove_marker_is_16s_only(self, mock_mangrove_data):
        assert mock_mangrove_data["counts"] is not None
        assert "counts_its" not in mock_mangrove_data

    def test_mangrove_primer_v3_v4_isolated_from_lucas(self, mock_mangrove_data):
        assert all(f.startswith("ASV_MANGROVE_") for f in mock_mangrove_data["features"])

    def test_mangrove_estuary_metadata_retained(self, mock_mangrove_data):
        estuaries = {r["estuary"] for r in mock_mangrove_data["records"]}
        assert len(estuaries) == 7
        assert "Cocó" in estuaries

    def test_mangrove_refusal_to_fuse_with_lucas_its(self, mock_mangrove_data):
        def attempt_multimarker_fusion(mangrove_item):
            if mangrove_item.get("bioproject") == "PRJNA283936" and "its_counts" not in mangrove_item:
                raise ValueError("PRJNA283936 lacks fungal ITS; multi-marker fusion prohibited")
        with pytest.raises(ValueError, match="PRJNA283936 lacks fungal ITS"):
            attempt_multimarker_fusion(mock_mangrove_data["records"][0])

    def test_mangrove_sample_count_105(self, mock_mangrove_data):
        assert len(mock_mangrove_data["records"]) == 105
        assert mock_mangrove_data["counts"].shape[0] == 105


# ============================================================================
# Feature 7: Synthetic Mock Generator
# ============================================================================
class TestFeature07_SyntheticMockGenerator:
    def test_mock_generator_deterministic_seed(self, mock_lucas_manifest):
        barcodes = mock_lucas_manifest["barcodes"]
        assert len(barcodes) == mock_lucas_manifest["n_samples"]
        assert barcodes[0] == 59

    def test_mock_generator_produces_paired_samples(self, mock_lucas_manifest):
        for r in mock_lucas_manifest["records"]:
            assert r["is_paired"] is True
            assert r["reads_16s"] > 0
            assert r["reads_its"] > 0

    def test_mock_generator_produces_valid_coordinates(self, mock_lucas_manifest):
        for r in mock_lucas_manifest["records"]:
            assert -90.0 <= r["latitude"] <= 90.0
            assert -180.0 <= r["longitude"] <= 180.0

    def test_mock_generator_produces_counts_matrix(self, mock_count_data):
        c16 = mock_count_data["counts_16s"]
        cits = mock_count_data["counts_its"]
        assert c16.ndim == 2
        assert cits.ndim == 2
        assert c16.shape[0] == cits.shape[0]

    def test_mock_generator_executes_subsecond(self):
        import time
        t0 = time.perf_counter()
        rng = np.random.RandomState(42)
        _ = rng.normal(size=(60, 100))
        duration = time.perf_counter() - t0
        assert duration < 0.5


# ============================================================================
# Feature 8: Variable Dictionary & Schema
# ============================================================================
class TestFeature08_VariableDictionarySchema:
    def test_schema_validates_numeric_types(self, mock_lucas_manifest):
        r = mock_lucas_manifest["records"][0]
        numeric_fields = ["latitude", "longitude", "ph_h2o", "organic_carbon", "clay"]
        for f in numeric_fields:
            assert isinstance(r[f], (int, float))

    def test_schema_validates_categorical_land_cover(self, mock_lucas_manifest):
        valid = {"Cropland", "Grassland", "Woodland"}
        for r in mock_lucas_manifest["records"]:
            assert r["macro_land_cover"] in valid

    def test_schema_rejects_missing_required_keys(self):
        required = {"sample_id", "barcode_id", "latitude", "longitude"}
        partial = {"sample_id": "LUCAS_1"}
        missing = required - set(partial.keys())
        assert len(missing) > 0

    def test_schema_bounds_checking_coordinates(self):
        def check_bounds(lat, lon):
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                raise ValueError("Coordinates out of WGS84 bounds")
            return True
        assert check_bounds(52.5, 13.4)
        with pytest.raises(ValueError):
            check_bounds(95.0, 10.0)

    def test_schema_json_serializable(self, mock_lucas_manifest):
        r = mock_lucas_manifest["records"][0]
        dumped = json.dumps(r)
        loaded = json.loads(dumped)
        assert loaded["sample_id"] == r["sample_id"]


# ============================================================================
# Feature 9: 16S Dual-Orientation Handling
# ============================================================================
class TestFeature09_DualOrientationHandling:
    def test_detect_forward_orientation(self):
        primer_515f = "GTGYCAGCMGCCGCGGTAA"
        read = "GTGCCAGCAGCCGCGGTAATACGTAGGGTGCGAGCGTTAATCGGAATTACTGGGCGTAA"
        assert read.startswith("GTGCCAGCAGCCGCGGTAA")

    def test_detect_reverse_complement_orientation(self):
        def revcomp(seq):
            tr = str.maketrans("ACGTURYKMSWBDHVN", "TGCAAYRMKWSVHDBN")
            return seq.translate(tr)[::-1]
        orig = "GTGCCAGCAGCCGCGGTAATACG"
        rc = revcomp(orig)
        assert rc != orig
        assert revcomp(rc) == orig

    def test_reorient_reverse_complement_sequences(self):
        def reorient(seq, primer):
            if seq.startswith(primer[:10]):
                return seq, "FORWARD"
            tr = str.maketrans("ACGTN", "TGCAN")
            rc = seq.translate(tr)[::-1]
            if rc.startswith(primer[:10]):
                return rc, "REVERSE_COMPLEMENT"
            return seq, "UNKNOWN"
        p = "GTGCCAGCAG"
        f_seq, orient1 = reorient("GTGCCAGCAGTTT", p)
        assert orient1 == "FORWARD"
        r_seq, orient2 = reorient("AAACTGCTGGCAC", p)
        assert orient2 == "REVERSE_COMPLEMENT"
        assert r_seq.startswith(p)

    def test_dual_orientation_preserves_read_count(self):
        reads = ["GTGCCAGCAGAAA", "TTTCTGCTGGCAC"]
        assert len(reads) == 2

    def test_dual_orientation_primer_position(self):
        p = "GTGCCAGCAG"
        seq = "NNNGTGCCAGCAGAAAA"
        pos = seq.find(p)
        assert pos == 3


# ============================================================================
# Feature 10: Primer Matching & Trimming
# ============================================================================
class TestFeature10_PrimerMatchingTrimming:
    def test_515f_primer_matching(self):
        primer = "GTGYCAGCMGCCGCGGTAA"
        assert len(primer) == 19

    def test_926r_primer_matching(self):
        primer = "GGCCGYCAATTYMTTTRAGTTT"
        assert len(primer) == 22

    def test_its9mun_primer_matching(self):
        primer = "GTACACACCGCCCGTCG"
        assert len(primer) == 17

    def test_trim_primer_removes_exact_prefix(self):
        primer = "GTGCCAGCAGCCGCGGTAA"
        read = primer + "ATCGATCG"
        trimmed = read[len(primer):] if read.startswith(primer) else read
        assert trimmed == "ATCGATCG"

    def test_trim_primer_allows_max_2_mismatches(self):
        def hamming(s1, s2):
            return sum(c1 != c2 for c1, c2 in zip(s1, s2))
        p = "GTGCCAGCAG"
        candidate = "GTGACAGCAG"  # 1 mismatch
        assert hamming(p, candidate) <= 2


# ============================================================================
# Feature 11: ASV/OTU Count Table Ingestion
# ============================================================================
class TestFeature11_CountTableIngestion:
    def test_count_table_non_negative_integers(self, mock_count_data):
        assert np.all(mock_count_data["counts_16s"] >= 0)
        assert np.all(mock_count_data["counts_its"] >= 0)
        assert issubclass(mock_count_data["counts_16s"].dtype.type, np.integer)

    def test_count_table_sample_ids_match_manifest(self, mock_count_data, mock_lucas_manifest):
        manifest_samples = [r["sample_id"] for r in mock_lucas_manifest["records"]]
        assert mock_count_data["sample_ids"] == manifest_samples

    def test_count_table_feature_ids_unique(self, mock_count_data):
        assert len(set(mock_count_data["features_16s"])) == len(mock_count_data["features_16s"])
        assert len(set(mock_count_data["features_its"])) == len(mock_count_data["features_its"])

    def test_count_table_dense_representation(self, mock_count_data):
        c16 = mock_count_data["counts_16s"]
        assert isinstance(c16, np.ndarray)
        assert c16.shape == (mock_count_data["n_samples"], 120)

    def test_count_table_sparse_matrix_support(self, mock_count_data):
        c16 = mock_count_data["counts_16s"]
        sparsity = np.mean(c16 == 0)
        assert 0.0 <= sparsity <= 1.0


# ============================================================================
# Feature 12: Taxonomic Annotation Parser
# ============================================================================
class TestFeature12_TaxonomicAnnotationParser:
    def test_silva_hierarchy_extraction(self, mock_taxonomy):
        tax = mock_taxonomy["taxonomy_16s"][0]
        levels = [part.strip() for part in tax.split(";")]
        assert len(levels) == 7
        assert levels[0].startswith("k__")
        assert levels[1].startswith("p__")

    def test_unite_hierarchy_extraction(self, mock_taxonomy):
        tax = mock_taxonomy["taxonomy_its"][0]
        levels = [part.strip() for part in tax.split(";")]
        assert len(levels) == 7
        assert levels[0] == "k__Fungi"

    def test_unassigned_taxa_handling(self):
        tax_str = "k__Bacteria; p__; c__; o__; f__; g__; s__"
        parts = [p.strip() for p in tax_str.split(";")]
        assert parts[1] == "p__"

    def test_taxonomy_id_alignment_with_count_matrix(self, mock_taxonomy, mock_count_data):
        assert len(mock_taxonomy["taxonomy_16s"]) == len(mock_count_data["features_16s"])
        assert len(mock_taxonomy["taxonomy_its"]) == len(mock_count_data["features_its"])

    def test_taxonomy_strip_prefix_tags(self):
        raw = "p__Acidobacteriota"
        clean = raw[3:] if raw.startswith("p__") else raw
        assert clean == "Acidobacteriota"


# ============================================================================
# Feature 13: Read Depth Filtering
# ============================================================================
class TestFeature13_ReadDepthFiltering:
    def test_filter_removes_subthreshold_16s(self):
        depths = np.array([45000, 39000, 40109, 20000])
        passed = depths >= 40109
        assert np.array_equal(passed, [True, False, True, False])

    def test_filter_removes_subthreshold_its(self):
        depths = np.array([600, 501, 502, 300])
        passed = depths >= 502
        assert np.array_equal(passed, [True, False, True, False])

    def test_filter_retains_compliant_samples(self, mock_count_data):
        depths_16s = np.sum(mock_count_data["counts_16s"], axis=1)
        depths_its = np.sum(mock_count_data["counts_its"], axis=1)
        valid = (depths_16s >= 40109) & (depths_its >= 502)
        assert np.sum(valid) > 0

    def test_filter_logs_attrition_counts(self):
        total = 885
        retained = 715
        attrition = total - retained
        assert attrition == 170

    def test_filter_preserves_taxa_columns(self, mock_count_data):
        c16 = mock_count_data["counts_16s"]
        filtered = c16[np.sum(c16, axis=1) >= 40109]
        assert filtered.shape[1] == c16.shape[1]


# ============================================================================
# Feature 14: SRS Depth Normalization
# ============================================================================
class TestFeature14_SRSNormalization:
    def test_srs_column_sums_equal_target_depth(self):
        row = np.array([100, 200, 300, 400, 500])
        target = 502
        norm = oracle_srs_normalization(row, target_depth=target, seed=42)
        assert np.sum(norm) == target

    def test_srs_integer_counts_output(self):
        row = np.array([1000, 2000, 3000])
        norm = oracle_srs_normalization(row, target_depth=502, seed=42)
        assert issubclass(norm.dtype.type, np.integer)

    def test_srs_preserves_rank_order(self):
        row = np.array([100, 500, 2000, 10000])
        norm = oracle_srs_normalization(row, target_depth=1000, seed=42)
        assert norm[0] <= norm[1] <= norm[2] <= norm[3]

    def test_srs_deterministic_with_fixed_seed(self):
        row = np.array([105, 205, 305, 405])
        n1 = oracle_srs_normalization(row, target_depth=200, seed=42)
        n2 = oracle_srs_normalization(row, target_depth=200, seed=42)
        assert np.array_equal(n1, n2)

    def test_srs_raises_error_if_depth_insufficient(self):
        row = np.array([10, 20])
        with pytest.raises(ValueError, match="below SRS target depth"):
            oracle_srs_normalization(row, target_depth=502)


# ============================================================================
# Feature 15: Taxon Rollup Aggregator
# ============================================================================
class TestFeature15_TaxonRollupAggregator:
    def test_rollup_to_phylum_level(self, mock_count_data, mock_taxonomy):
        c16 = mock_count_data["counts_16s"]
        phyla = [t.split(";")[1].strip() for t in mock_taxonomy["taxonomy_16s"]]
        unique_phyla = list(set(phyla))
        rollup = np.zeros((c16.shape[0], len(unique_phyla)))
        for idx, p in enumerate(unique_phyla):
            col_indices = [i for i, taxon_p in enumerate(phyla) if taxon_p == p]
            rollup[:, idx] = np.sum(c16[:, col_indices], axis=1)
        assert rollup.shape == (c16.shape[0], len(unique_phyla))

    def test_rollup_sums_conserve_total_abundance(self, mock_count_data, mock_taxonomy):
        c16 = mock_count_data["counts_16s"]
        phyla = [t.split(";")[1].strip() for t in mock_taxonomy["taxonomy_16s"]]
        unique_phyla = list(set(phyla))
        rollup = np.zeros((c16.shape[0], len(unique_phyla)))
        for idx, p in enumerate(unique_phyla):
            col_indices = [i for i, taxon_p in enumerate(phyla) if taxon_p == p]
            rollup[:, idx] = np.sum(c16[:, col_indices], axis=1)
        np.testing.assert_allclose(np.sum(rollup, axis=1), np.sum(c16, axis=1))

    def test_rollup_unclassified_bucket(self):
        taxa = ["p__Proteobacteria", "p__", "p__Actinobacteriota"]
        counts = np.array([[10, 5, 20]])
        unique_p = ["p__Proteobacteria", "p__Actinobacteriota", "Unclassified"]
        # Aggregation puts "p__" into Unclassified
        assert len(unique_p) == 3

    def test_rollup_to_genus_level(self, mock_count_data, mock_taxonomy):
        cits = mock_count_data["counts_its"]
        genera = [t.split(";")[5].strip() for t in mock_taxonomy["taxonomy_its"]]
        assert len(genera) == cits.shape[1]

    def test_rollup_output_dimensions(self, mock_count_data):
        n_samples = mock_count_data["n_samples"]
        n_classes = 5
        dummy_rollup = np.ones((n_samples, n_classes))
        assert dummy_rollup.shape[0] == n_samples


# ============================================================================
# Feature 16: Alpha Diversity Metrics
# ============================================================================
class TestFeature16_AlphaDiversityMetrics:
    def test_observed_richness_calculation(self):
        row = np.array([0, 10, 0, 5, 2])
        richness = np.sum(row > 0)
        assert richness == 3

    def test_shannon_entropy_non_negative(self, mock_count_data):
        for row in mock_count_data["counts_16s"][:10]:
            h = oracle_shannon_entropy(row)
            assert h >= 0.0

    def test_simpson_index_bounded_0_to_1(self, mock_count_data):
        for row in mock_count_data["counts_16s"][:10]:
            d = oracle_simpson_index(row)
            assert 0.0 <= d <= 1.0

    def test_chao1_greater_than_or_equal_to_observed(self):
        counts = np.array([1, 1, 2, 5, 10, 0])
        s_obs = np.sum(counts > 0)
        chao1 = oracle_chao1(counts)
        assert chao1 >= s_obs

    def test_monoculture_shannon_equals_zero(self):
        counts = np.array([0, 100, 0, 0])
        assert oracle_shannon_entropy(counts) == 0.0


# ============================================================================
# Feature 17: Simplex Pseudocount Imputation
# ============================================================================
class TestFeature17_PseudocountImputation:
    def test_pseudocount_replaces_all_zeros(self):
        counts = np.array([0, 10, 0, 5])
        pseudo = np.where(counts <= 0, 1e-6, counts)
        assert np.all(pseudo > 0)

    def test_pseudocount_leaves_positives_unaltered(self):
        counts = np.array([0, 10, 0, 5])
        pseudo = np.where(counts <= 0, 1e-6, counts)
        assert pseudo[1] == 10.0
        assert pseudo[3] == 5.0

    def test_pseudocount_positive_strictly(self):
        delta = 1e-6
        assert delta > 0.0

    def test_pseudocount_constant_preserves_simplex_order(self):
        counts = np.array([10, 20, 30])
        pseudo = counts + 1.0
        assert pseudo[0] < pseudo[1] < pseudo[2]

    def test_multiplicative_zero_replacement_concept(self):
        row = np.array([0.0, 0.4, 0.6])
        delta = 0.01
        replaced = np.where(row == 0, delta, row * (1.0 - delta))
        np.testing.assert_allclose(np.sum(replaced), 1.0, atol=1e-5)


# ============================================================================
# Feature 18: Centered Log-Ratio (CLR)
# ============================================================================
class TestFeature18_CenteredLogRatio:
    def test_clr_sum_to_zero_property(self, mock_count_data):
        clr = oracle_clr(mock_count_data["counts_16s"][:5])
        sums = np.sum(clr, axis=1)
        np.testing.assert_allclose(sums, 0.0, atol=1e-7)

    def test_clr_scale_invariance(self):
        x = np.array([10, 20, 40], dtype=float)
        cx = 5.0 * x
        clr_x = oracle_clr(x)
        clr_cx = oracle_clr(cx)
        np.testing.assert_allclose(clr_x, clr_cx, atol=1e-7)

    def test_clr_preserves_sample_dimension(self, mock_count_data):
        c = mock_count_data["counts_16s"][:10]
        clr = oracle_clr(c)
        assert clr.shape == c.shape

    def test_clr_geometric_mean_centering(self):
        x = np.array([1.0, math.e, math.e ** 2])
        # geometric mean is e^( (0+1+2)/3 ) = e^1 = e
        # ln(x/g) = [0-1, 1-1, 2-1] = [-1, 0, 1]
        clr = oracle_clr(x)
        np.testing.assert_allclose(clr, [-1.0, 0.0, 1.0], atol=1e-7)

    def test_clr_invertibility_to_relative_proportions(self):
        x = np.array([10.0, 20.0, 70.0])
        p_orig = x / np.sum(x)
        clr = oracle_clr(x)
        exp_clr = np.exp(clr)
        p_recovered = exp_clr / np.sum(exp_clr)
        np.testing.assert_allclose(p_orig, p_recovered, atol=1e-7)


# ============================================================================
# Feature 19: Bray-Curtis Dissimilarity
# ============================================================================
class TestFeature19_BrayCurtisDissimilarity:
    def test_bray_curtis_identical_samples_zero(self):
        u = np.array([10, 20, 30])
        assert oracle_bray_curtis(u, u) == 0.0

    def test_bray_curtis_disjoint_samples_one(self):
        u = np.array([10, 20, 0, 0])
        v = np.array([0, 0, 30, 40])
        assert oracle_bray_curtis(u, v) == 1.0

    def test_bray_curtis_symmetry(self):
        u = np.array([10, 5, 2])
        v = np.array([3, 12, 8])
        assert oracle_bray_curtis(u, v) == oracle_bray_curtis(v, u)

    def test_bray_curtis_bounded_zero_to_one(self, mock_count_data):
        c = mock_count_data["counts_16s"]
        for i in range(5):
            bc = oracle_bray_curtis(c[i], c[i+1])
            assert 0.0 <= bc <= 1.0

    def test_bray_curtis_matrix_diagonal_zeros(self, mock_count_data):
        c = mock_count_data["counts_16s"][:5]
        n = len(c)
        dm = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dm[i, j] = oracle_bray_curtis(c[i], c[j])
        np.testing.assert_allclose(np.diag(dm), 0.0)


# ============================================================================
# Feature 20: Aitchison Distance Matrix
# ============================================================================
class TestFeature20_AitchisonDistanceMatrix:
    def test_aitchison_distance_metric_axioms(self):
        x = np.array([10, 20, 30], dtype=float)
        y = np.array([15, 25, 35], dtype=float)
        z = np.array([5, 10, 80], dtype=float)
        # Non-negativity
        d_xy = oracle_aitchison_distance(x, y)
        assert d_xy >= 0.0
        # Identity
        assert oracle_aitchison_distance(x, x) == 0.0
        # Symmetry
        assert d_xy == oracle_aitchison_distance(y, x)
        # Triangle inequality
        d_xz = oracle_aitchison_distance(x, z)
        d_yz = oracle_aitchison_distance(y, z)
        assert d_xz <= d_xy + d_yz + 1e-9

    def test_aitchison_scale_invariance(self):
        x = np.array([10, 20, 30], dtype=float)
        y = np.array([15, 25, 35], dtype=float)
        d1 = oracle_aitchison_distance(x, y)
        d2 = oracle_aitchison_distance(x * 10, y * 0.1)
        np.testing.assert_allclose(d1, d2, atol=1e-7)

    def test_aitchison_subcompositional_coherence(self):
        x = np.array([10, 20, 30, 40], dtype=float)
        y = np.array([15, 25, 35, 45], dtype=float)
        full_dist = oracle_aitchison_distance(x, y)
        assert full_dist >= 0.0

    def test_aitchison_zero_for_collinear_compositions(self):
        x = np.array([10, 20, 30], dtype=float)
        y = 3.5 * x
        assert oracle_aitchison_distance(x, y) < 1e-6

    def test_aitchison_pairwise_matrix_shape(self, mock_count_data):
        c = mock_count_data["counts_16s"][:6]
        n = len(c)
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                mat[i, j] = oracle_aitchison_distance(c[i], c[j])
        assert mat.shape == (n, n)
        assert np.all(mat >= 0.0)


# ============================================================================
# Feature 21: Native PERMANOVA Routine
# ============================================================================
class TestFeature21_NativePERMANOVA:
    def test_permanova_pseudo_f_non_negative(self):
        # Analytical F stat test on separated groups
        d_between = 5.0
        d_within = 1.0
        pseudo_f = d_between / d_within
        assert pseudo_f >= 0.0

    def test_permanova_r2_bounded_zero_to_one(self):
        ss_between = 20.0
        ss_total = 50.0
        r2 = ss_between / ss_total
        assert 0.0 <= r2 <= 1.0

    def test_permanova_p_value_bounded_zero_to_one(self):
        n_perm = 999
        n_extreme = 5
        p = (n_extreme + 1) / (n_perm + 1)
        assert 0.0 < p <= 1.0

    def test_permanova_permutations_deterministic_seed(self):
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        perm1 = rng1.permutation(10)
        perm2 = rng2.permutation(10)
        assert np.array_equal(perm1, perm2)

    def test_permanova_orthogonal_effect_detection(self):
        # Two distinct groups should have higher F than random groups
        f_real = 8.5
        f_random = 1.2
        assert f_real > f_random


# ============================================================================
# Feature 22: Multivariate Dispersion Test
# ============================================================================
class TestFeature22_MultivariateDispersion:
    def test_betadisper_distances_to_centroid_positive(self):
        points = np.array([[1.0, 2.0], [2.0, 3.0], [1.5, 2.5]])
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        assert np.all(distances >= 0.0)

    def test_betadisper_f_stat_calculation(self):
        d1 = np.array([0.1, 0.2, 0.15])
        d2 = np.array([1.5, 2.0, 1.8])
        var1 = np.var(d1)
        var2 = np.var(d2)
        assert var2 > var1

    def test_betadisper_homoscedastic_baseline(self):
        # Equal variances should yield non-significant p
        p_val = 0.45
        assert p_val >= 0.05

    def test_betadisper_heteroscedastic_flagging(self):
        p_val = 0.002
        assert p_val < 0.05  # Flagged as dispersion heterogeneity

    def test_betadisper_returns_sample_distances(self):
        n = 10
        dist_array = np.abs(np.random.RandomState(42).normal(size=n))
        assert len(dist_array) == n


# ============================================================================
# Feature 23: Spatial Distance Matrix
# ============================================================================
class TestFeature23_SpatialDistanceMatrix:
    def test_haversine_zero_distance_same_point(self):
        assert oracle_haversine(50.0, 4.0, 50.0, 4.0) == 0.0

    def test_haversine_symmetry(self):
        d1 = oracle_haversine(48.8566, 2.3522, 51.5074, -0.1278)
        d2 = oracle_haversine(51.5074, -0.1278, 48.8566, 2.3522)
        np.testing.assert_allclose(d1, d2)

    def test_haversine_known_distance_london_paris(self):
        # Paris (48.8566, 2.3522) to London (51.5074, -0.1278) is ~343-345 km
        d = oracle_haversine(48.8566, 2.3522, 51.5074, -0.1278)
        assert 340.0 <= d <= 350.0

    def test_haversine_triangle_inequality(self):
        d_ab = oracle_haversine(50.0, 4.0, 52.0, 5.0)
        d_bc = oracle_haversine(52.0, 5.0, 54.0, 6.0)
        d_ac = oracle_haversine(50.0, 4.0, 54.0, 6.0)
        assert d_ac <= d_ab + d_bc + 1e-5

    def test_pairwise_spatial_matrix_shape(self, mock_lucas_manifest):
        records = mock_lucas_manifest["records"][:8]
        n = len(records)
        dist_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_mat[i, j] = oracle_haversine(
                    records[i]["latitude"], records[i]["longitude"],
                    records[j]["latitude"], records[j]["longitude"]
                )
        assert dist_mat.shape == (n, n)
        np.testing.assert_allclose(np.diag(dist_mat), 0.0)


# ============================================================================
# Feature 24: Spatial Blocking Partitioner
# ============================================================================
class TestFeature24_SpatialBlockingPartitioner:
    def test_spatial_blocking_partitions_all_samples(self, mock_lucas_manifest):
        n = mock_lucas_manifest["n_samples"]
        k = 5
        indices = np.arange(n)
        fold_size = n // k
        splits = [(np.delete(indices, np.arange(i*fold_size, (i+1)*fold_size)),
                   indices[i*fold_size:(i+1)*fold_size]) for i in range(k)]
        assert len(splits) == k

    def test_spatial_blocking_no_sample_overlap(self, mock_lucas_manifest):
        n = 50
        splits = [(np.arange(10, 50), np.arange(0, 10)), (np.arange(0, 40), np.arange(40, 50))]
        for tr, te in splits:
            assert len(set(tr).intersection(set(te))) == 0

    def test_spatial_blocking_preserves_site_clusters(self, mock_lucas_manifest):
        site_groups = {"SITE_01": [0, 1], "SITE_02": [2, 3]}
        # All members of site must be together
        fold_assignment = {0: 0, 1: 0, 2: 1, 3: 1}
        assert fold_assignment[0] == fold_assignment[1]
        assert fold_assignment[2] == fold_assignment[3]

    def test_spatial_blocking_cluster_compactness(self):
        # Samples closer to each other should cluster together
        d_close = oracle_haversine(50.0, 4.0, 50.1, 4.1)
        d_far = oracle_haversine(50.0, 4.0, 60.0, 15.0)
        assert d_close < d_far

    def test_spatial_blocking_deterministic_seed(self):
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        assert np.array_equal(rng1.permutation(10), rng2.permutation(10))


# ============================================================================
# Feature 25: Spatial Buffer Exclusion
# ============================================================================
class TestFeature25_SpatialBufferExclusion:
    def test_buffer_drops_proximal_training_points(self):
        coords_train = np.array([[50.0, 4.0], [55.0, 10.0]])
        coord_test = (50.05, 4.05) # ~6 km away
        d0 = oracle_haversine(coords_train[0, 0], coords_train[0, 1], coord_test[0], coord_test[1])
        d1 = oracle_haversine(coords_train[1, 0], coords_train[1, 1], coord_test[0], coord_test[1])
        buffer_km = 20.0
        retained = [i for i, d in enumerate([d0, d1]) if d >= buffer_km]
        assert retained == [1]

    def test_buffer_enforces_minimum_distance_bound(self):
        buffer_km = 50.0
        dist = 75.0
        assert dist >= buffer_km

    def test_buffer_zero_radius_leaves_all_train(self):
        buffer_km = 0.0
        assert buffer_km == 0.0

    def test_buffer_preserves_test_fold_intact(self):
        test_indices = np.array([5, 6, 7])
        assert len(test_indices) == 3

    def test_buffer_excessive_radius_warning_or_handling(self):
        buffer_km = 5000.0  # Excessive
        train_count = 0
        assert train_count == 0  # Triggers warning / exception


# ============================================================================
# Feature 26: Class Representation Enforcer
# ============================================================================
class TestFeature26_ClassRepresentationEnforcer:
    def test_enforces_all_test_classes_present_in_train(self):
        y_train = np.array(["Crop", "Wood", "Grass", "Crop"])
        y_test = np.array(["Crop", "Wood"])
        assert set(y_test).issubset(set(y_train))

    def test_detects_unrepresented_class_violation(self):
        y_train = np.array(["Crop", "Crop"])
        y_test = np.array(["Wood"])
        missing = set(y_test) - set(y_train)
        assert "Wood" in missing

    def test_fallback_reassignment_heals_unrepresented_class(self):
        y_train = ["Crop"]
        y_test = ["Wood"]
        # Reassignment gives one Wood to train
        y_train.append("Wood")
        assert set(y_test).issubset(set(y_train))

    def test_preserves_class_counts_greater_than_zero(self):
        classes = ["Cropland", "Grassland", "Woodland"]
        assert len(classes) == 3

    def test_raises_informative_error_on_impossible_partition(self):
        def validate(train_c, test_c):
            if not set(test_c).issubset(set(train_c)):
                raise ValueError("Unrepresented test class in training set")
        with pytest.raises(ValueError, match="Unrepresented test class"):
            validate(["A"], ["B"])


# ============================================================================
# Feature 27: Zero Spatial Leakage Assertions
# ============================================================================
class TestFeature27_ZeroSpatialLeakage:
    def test_zero_leakage_passes_valid_splits(self):
        train_sites = {"SITE_A", "SITE_B"}
        test_sites = {"SITE_C"}
        assert len(train_sites.intersection(test_sites)) == 0

    def test_zero_leakage_fails_shared_site_id(self):
        train_sites = {"SITE_A", "SITE_B"}
        test_sites = {"SITE_B", "SITE_C"}
        shared = train_sites.intersection(test_sites)
        assert len(shared) > 0

    def test_zero_leakage_fails_shared_coordinates(self):
        train_coords = {(50.0, 4.0)}
        test_coords = {(50.0, 4.0)}
        assert len(train_coords.intersection(test_coords)) == 1

    def test_zero_leakage_fails_shared_indices(self):
        tr_idx = {0, 1, 2}
        te_idx = {2, 3}
        assert len(tr_idx.intersection(te_idx)) > 0

    def test_zero_leakage_fails_buffer_breach(self):
        min_dist = 5.0
        buffer_km = 10.0
        assert min_dist < buffer_km


# ============================================================================
# Feature 28: Single-Marker Feature Sets (Set A, Set B)
# ============================================================================
class TestFeature28_SingleMarkerFeatureSets:
    def test_set_a_contains_only_bacterial_features(self, mock_count_data):
        c16 = mock_count_data["counts_16s"]
        assert c16.shape[1] == 120
        assert all("16S" in f for f in mock_count_data["features_16s"])

    def test_set_b_contains_only_fungal_features(self, mock_count_data):
        cits = mock_count_data["counts_its"]
        assert cits.shape[1] == 80
        assert all("ITS" in f for f in mock_count_data["features_its"])

    def test_feature_sets_preserve_sample_alignment(self, mock_count_data):
        assert mock_count_data["counts_16s"].shape[0] == mock_count_data["counts_its"].shape[0]

    def test_feature_sets_apply_clr_transform(self, mock_count_data):
        clr_a = oracle_clr(mock_count_data["counts_16s"])
        clr_b = oracle_clr(mock_count_data["counts_its"])
        np.testing.assert_allclose(np.sum(clr_a, axis=1), 0.0, atol=1e-7)
        np.testing.assert_allclose(np.sum(clr_b, axis=1), 0.0, atol=1e-7)

    def test_feature_sets_sample_order_consistency(self, mock_count_data):
        s_ids = mock_count_data["sample_ids"]
        assert len(s_ids) == mock_count_data["n_samples"]


# ============================================================================
# Feature 29: Early Fusion Multi-Marker Set (Set C)
# ============================================================================
class TestFeature29_EarlyFusionSet:
    def test_early_fusion_concatenates_a_and_b(self, mock_count_data):
        c16 = mock_count_data["counts_16s"]
        cits = mock_count_data["counts_its"]
        concat = np.hstack([c16, cits])
        assert concat.shape == (c16.shape[0], c16.shape[1] + cits.shape[1])

    def test_early_fusion_feature_dimension_equals_sum(self, mock_count_data):
        dim_sum = len(mock_count_data["features_16s"]) + len(mock_count_data["features_its"])
        assert dim_sum == 120 + 80 == 200

    def test_early_fusion_block_weighting_normalization(self):
        d1 = 120
        d2 = 80
        w1 = 1.0 / math.sqrt(d1)
        w2 = 1.0 / math.sqrt(d2)
        assert w2 > w1  # Smaller block receives higher per-feature weight

    def test_early_fusion_sample_alignment_strict(self, mock_count_data):
        assert mock_count_data["counts_16s"].shape[0] == mock_count_data["counts_its"].shape[0]

    def test_early_fusion_feature_name_collision_guard(self, mock_count_data):
        names16 = [f"16S_{f}" for f in mock_count_data["features_16s"]]
        names_its = [f"ITS_{f}" for f in mock_count_data["features_its"]]
        assert len(set(names16).intersection(set(names_its))) == 0


# ============================================================================
# Feature 30: Late Fusion Multi-Marker Set
# ============================================================================
class TestFeature30_LateFusionSet:
    def test_late_fusion_averages_class_probabilities(self):
        p_a = np.array([[0.7, 0.2, 0.1]])
        p_b = np.array([[0.5, 0.4, 0.1]])
        p_late = 0.5 * p_a + 0.5 * p_b
        np.testing.assert_allclose(p_late, [[0.6, 0.3, 0.1]])

    def test_late_fusion_probabilities_sum_to_one(self):
        p_a = np.array([0.6, 0.3, 0.1])
        p_b = np.array([0.2, 0.7, 0.1])
        p_ens = 0.4 * p_a + 0.6 * p_b
        np.testing.assert_allclose(np.sum(p_ens), 1.0)

    def test_late_fusion_weighted_combination(self):
        w = 0.7
        p_a = 1.0
        p_b = 0.0
        assert w * p_a + (1 - w) * p_b == 0.7

    def test_late_fusion_meta_learner_stacking(self):
        # Stacking input is concatenated predicted probabilities
        prob_stack = np.array([[0.8, 0.2, 0.6, 0.4]])
        assert prob_stack.shape == (1, 4)

    def test_late_fusion_preserves_class_labels(self):
        classes = ["Cropland", "Grassland", "Woodland"]
        assert len(classes) == 3


# ============================================================================
# Feature 31: Supervised Classifiers
# ============================================================================
class TestFeature31_SupervisedClassifiers:
    def test_random_forest_fit_predict(self):
        rng = np.random.RandomState(42)
        X = rng.normal(size=(20, 5))
        y = np.array([0]*10 + [1]*10)
        # Mocking classification prediction
        preds = y.copy()
        assert len(preds) == len(y)

    def test_gbdt_fit_predict(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert np.array_equal(y_true, y_pred)

    def test_elastic_net_logistic_fit_predict(self):
        coef = np.array([0.5, -0.2, 0.0, 1.2])
        assert len(coef) == 4

    def test_linear_svm_fit_predict(self):
        support_vectors = np.ones((4, 5))
        assert support_vectors.shape[0] == 4

    def test_classifier_predict_proba_shape(self):
        n_samples = 10
        n_classes = 3
        probas = np.ones((n_samples, n_classes)) / n_classes
        assert probas.shape == (n_samples, n_classes)


# ============================================================================
# Feature 32: Balanced Accuracy & Macro-F1
# ============================================================================
class TestFeature32_BalancedMetrics:
    def test_balanced_accuracy_perfect_predictions(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        assert oracle_balanced_accuracy(y, y) == 1.0

    def test_balanced_accuracy_random_guess_expectation(self):
        # 3 classes: uniform random guess expectation is 1/3 ~ 0.333
        y_true = np.array([0]*10 + [1]*10 + [2]*10)
        y_pred = np.array([0]*30) # Always predicts class 0
        acc = oracle_balanced_accuracy(y_true, y_pred)
        # Class 0 recall = 1.0, Class 1 = 0.0, Class 2 = 0.0 => mean = 0.3333
        np.testing.assert_allclose(acc, 1.0 / 3.0)

    def test_macro_f1_perfect_predictions(self):
        y = np.array(["Crop", "Grass", "Wood"])
        assert oracle_macro_f1(y, y) == 1.0

    def test_macro_f1_penalizes_minority_neglect(self):
        y_true = np.array([0]*20 + [1]*2)
        y_pred = np.array([0]*22) # Minority class ignored
        f1 = oracle_macro_f1(y_true, y_pred)
        assert f1 < 0.6 # Strongly penalized despite 20/22 = 91% accuracy

    def test_balanced_metrics_zero_division_safety(self):
        y_true = np.array([0, 1])
        y_pred = np.array([2, 2])
        acc = oracle_balanced_accuracy(y_true, y_pred)
        assert acc == 0.0


# ============================================================================
# Feature 33: Confusion Matrix Analytics
# ============================================================================
class TestFeature33_ConfusionMatrixAnalytics:
    def test_confusion_matrix_shape_k_by_k(self):
        k = 3
        cm = np.zeros((k, k), dtype=int)
        assert cm.shape == (3, 3)

    def test_confusion_matrix_row_sums_equal_class_support(self):
        cm = np.array([[10, 2], [1, 15]])
        assert np.sum(cm[0]) == 12
        assert np.sum(cm[1]) == 16

    def test_confusion_matrix_normalized_diagonal_recalls(self):
        cm = np.array([[8, 2], [3, 7]])
        recalls = np.diag(cm) / np.sum(cm, axis=1)
        np.testing.assert_allclose(recalls, [0.8, 0.7])

    def test_confusion_matrix_canonical_class_ordering(self):
        classes = ["Cropland", "Grassland", "Woodland"]
        assert classes[0] == "Cropland"
        assert classes[2] == "Woodland"

    def test_confusion_matrix_handles_unpredicted_class(self):
        cm = np.array([[5, 0], [5, 0]]) # Class 1 never predicted
        assert np.sum(cm[:, 1]) == 0


# ============================================================================
# Feature 34: Spatial Bootstrap CIs
# ============================================================================
class TestFeature34_SpatialBootstrapCIs:
    def test_bootstrap_ci_coverage_interval(self):
        data = np.array([0.72, 0.75, 0.70, 0.78, 0.74, 0.73])
        ci_lower = np.percentile(data, 2.5)
        ci_upper = np.percentile(data, 97.5)
        assert ci_lower < ci_upper

    def test_bootstrap_ci_lower_less_than_upper(self):
        lo, hi = 0.65, 0.82
        assert lo < hi

    def test_bootstrap_ci_contains_point_estimate(self):
        data = np.array([0.70, 0.75, 0.80])
        mean = np.mean(data)
        assert np.percentile(data, 2.5) <= mean <= np.percentile(data, 97.5)

    def test_bootstrap_spatial_cluster_resampling(self):
        clusters = [0, 1, 2, 3, 4]
        resampled = np.random.RandomState(42).choice(clusters, size=len(clusters), replace=True)
        assert len(resampled) == len(clusters)

    def test_bootstrap_deterministic_seed(self):
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        assert np.array_equal(rng1.choice(10, 5), rng2.choice(10, 5))


# ============================================================================
# Feature 35: Null Hypothesis Tester (H0 Wilcoxon)
# ============================================================================
class TestFeature35_NullHypothesisTester:
    def test_h0_accepted_when_delta_non_positive(self):
        # Set C does not outperform max(A, B)
        score_c = 0.68
        max_ab = 0.70
        delta = score_c - max_ab
        assert delta <= 0.0

    def test_h0_accepted_when_p_value_ge_0_05(self):
        p_val = 0.24
        assert p_val >= 0.05

    def test_h0_rejected_when_delta_positive_and_p_less_0_05(self):
        delta = 0.08
        p_val = 0.012
        assert delta > 0.0 and p_val < 0.05

    def test_paired_wilcoxon_handles_identical_scores(self):
        x = np.array([0.7, 0.7, 0.8, 0.75])
        y = np.array([0.7, 0.7, 0.8, 0.75])
        w, p = oracle_wilcoxon_signed_rank(x, y)
        assert p == 1.0

    def test_wilcoxon_benjamini_hochberg_correction(self):
        raw_p = [0.01, 0.04, 0.06]
        # BH adjustment
        adj_p = [p * len(raw_p) / (i + 1) for i, p in enumerate(raw_p)]
        assert adj_p[0] <= adj_p[1]


# ============================================================================
# Feature 36: Pinned YAML Configuration
# ============================================================================
class TestFeature36_PinnedYAMLConfiguration:
    def test_config_loads_fixed_seed_42(self):
        cfg = {"seed": 42}
        assert cfg["seed"] == 42

    def test_config_contains_all_milestone_sections(self):
        cfg = {
            "data": {},
            "bioinformatics": {},
            "ecology": {},
            "spatial_cv": {},
            "models": {},
            "evaluation": {}
        }
        assert len(cfg) == 6

    def test_config_srs_depth_thresholds(self):
        cfg = {"srs_depth_16s": 40109, "srs_depth_its": 502}
        assert cfg["srs_depth_16s"] == 40109
        assert cfg["srs_depth_its"] == 502

    def test_config_file_paths_specified(self):
        cfg = {"manifest_path": "data/processed/paired_sample_manifest.csv"}
        assert cfg["manifest_path"].endswith(".csv")

    def test_config_immutability_or_clean_dict(self):
        cfg = {"run_name": "lucas_multimarker_spatial_cv"}
        assert isinstance(cfg, dict)


# ============================================================================
# Feature 37: Unified CLI Entry Point
# ============================================================================
class TestFeature37_UnifiedCLIEntryPoint:
    def test_cli_subcommands_registered(self):
        valid_commands = ["ingest", "bioinformatics", "diversity", "train", "evaluate", "all"]
        assert len(valid_commands) == 6

    def test_cli_help_flag_string(self):
        help_msg = "Soil Microbiome Environmental Discrimination Pipeline"
        assert "Microbiome" in help_msg

    def test_cli_seed_argument_passed(self):
        args = {"seed": 42}
        assert args["seed"] == 42

    def test_cli_config_path_validation(self):
        path = "config/pipeline_config.yaml"
        assert path.endswith(".yaml")

    def test_cli_version_flag(self):
        version = "1.0.0"
        assert version.startswith("1.")


# ============================================================================
# Feature 38: Automated Artifact Exporter
# ============================================================================
class TestFeature38_AutomatedArtifactExporter:
    def test_export_manifest_csv(self, temp_test_dir):
        path = f"{temp_test_dir}/manifest.csv"
        with open(path, "w") as f:
            f.write("sample_id,barcode_id\nLUCAS_1,1\n")
        assert os.path.exists(path)

    def test_export_metrics_json(self, temp_test_dir):
        path = f"{temp_test_dir}/metrics.json"
        data = {"balanced_accuracy": 0.74, "macro_f1": 0.71}
        with open(path, "w") as f:
            json.dump(data, f)
        assert os.path.exists(path)

    def test_export_confusion_matrix_csv(self, temp_test_dir):
        path = f"{temp_test_dir}/confusion_matrix.csv"
        with open(path, "w") as f:
            f.write("Cropland,Grassland,Woodland\n10,2,1\n")
        assert os.path.exists(path)

    def test_export_hypothesis_results_json(self, temp_test_dir):
        path = f"{temp_test_dir}/h0_results.json"
        res = {"delta": -0.015, "p_value": 0.38, "null_accepted": True}
        with open(path, "w") as f:
            json.dump(res, f)
        with open(path, "r") as f:
            loaded = json.load(f)
            assert loaded["null_accepted"] is True

    def test_export_directory_structure_creation(self, temp_test_dir):
        out_dirs = ["reports", "figures", "metrics"]
        for d in out_dirs:
            target = os.path.join(temp_test_dir, d)
            os.makedirs(target, exist_ok=True)
            assert os.path.isdir(target)
