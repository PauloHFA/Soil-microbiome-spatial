"""
tests/test_m1_adversarial_challenger.py
Milestone 1 Adversarial Challenger Stress-Test Suite.

Authored by m1_challenger_1.
Empirically stress-tests:
1. src/data/pairing_engine.py:
   - Mismatched barcode pairing rejection
   - select_best_run_per_sample tie-breaking (highest depth, accession tie-breaker, missing/NaN depths)
   - Canonical site ID mapping for duplicate sites (59/138, 102/601, 113/717, 116/258)
   - assert_no_resampled_site_leakage guard
2. src/data/sample_manifest.py:
   - Rejection of corrupt/out-of-bounds coordinates (lat, lon, NaN)
   - Rejection of invalid textures (texture sum > 105%)
   - Rejection of missing mandatory columns
   - Rejection of null fields in mandatory columns
   - Rejection of duplicate barcodes or sample IDs
   - Rejection of invalid land cover categories and SRA run accession strings
"""

import math
from typing import Set
import numpy as np
import pandas as pd
import pytest

from src.data.pairing_engine import (
    CANONICAL_SITE_ID_MAP,
    RESAMPLED_SITE_PAIRS,
    RESAMPLED_INITIAL_BARCODES,
    RESAMPLED_REPLACEMENT_BARCODES,
    DedupStrategy,
    PairingConfig,
    DataPairingError,
    SamplePairingViolationError,
    ResampledSiteCollisionError,
    assert_no_resampled_site_leakage,
    deduplicate_resampled_sites,
    enforce_strict_pairing,
    extract_barcode_id,
    pair_amplicon_runs,
    select_best_run_per_sample,
)
from src.data.sample_manifest import (
    MANDATORY_MANIFEST_COLUMNS,
    ManifestSchemaValidationError,
    validate_manifest_schema,
    load_paired_manifest,
    save_paired_manifest,
)
from src.data.mock_generator import MockDataGenerator, MockGeneratorConfig


# ============================================================================
# 1. Pairing Engine: Barcode Mismatch and Integrity Tests
# ============================================================================

class TestPairingEngineBarcodeIntegrity:
    """Stress-test barcode pairing integrity and cross-pairing prevention."""

    def test_completely_disjoint_barcodes_yield_empty_pairing(self):
        """Disjoint 16S and ITS barcodes must NEVER pair."""
        df_16s = pd.DataFrame([
            {"run_accession": "SRR16S_01", "sample_title": "LUCAS_10", "read_count": 50000, "marker": "16S"},
            {"run_accession": "SRR16S_02", "sample_title": "LUCAS_20", "read_count": 60000, "marker": "16S"},
        ])
        df_its = pd.DataFrame([
            {"run_accession": "SRRITS_01", "sample_title": "ITS__30", "read_count": 3000, "marker": "ITS"},
            {"run_accession": "SRRITS_02", "sample_title": "ITS__40", "read_count": 4000, "marker": "ITS"},
        ])
        paired, metrics = pair_amplicon_runs(df_16s, df_its)
        assert len(paired) == 0
        assert paired.empty
        assert set(metrics.orphan_16s_barcodes) == {10, 20}
        assert set(metrics.orphan_its_barcodes) == {30, 40}

    def test_mismatched_barcode_swapping_cannot_pair(self):
        """Cross-pairing where barcodes are swapped must only pair identical barcodes."""
        df_16s = pd.DataFrame([
            {"run_accession": "SRR16S_01", "sample_title": "LUCAS_1", "read_count": 50000, "marker": "16S"},
            {"run_accession": "SRR16S_02", "sample_title": "LUCAS_2", "read_count": 60000, "marker": "16S"},
        ])
        df_its = pd.DataFrame([
            {"run_accession": "SRRITS_01", "sample_title": "ITS__2", "read_count": 3000, "marker": "ITS"},
            {"run_accession": "SRRITS_02", "sample_title": "ITS__3", "read_count": 4000, "marker": "ITS"},
        ])
        paired, metrics = pair_amplicon_runs(df_16s, df_its)
        assert len(paired) == 1
        assert paired["barcode_id"].iloc[0] == 2
        assert paired["sra_16s_run"].iloc[0] == "SRR16S_02"
        assert paired["sra_its_run"].iloc[0] == "SRRITS_01"
        assert 1 in metrics.orphan_16s_barcodes
        assert 3 in metrics.orphan_its_barcodes

    def test_corrupt_or_unparseable_barcode_titles_ignored(self):
        """Sample titles without numbers or with invalid formats are ignored."""
        df_16s = pd.DataFrame([
            {"run_accession": "SRR16S_01", "sample_title": "NO_BARCODE_HERE", "read_count": 50000, "marker": "16S"},
            {"run_accession": "SRR16S_02", "sample_title": "", "read_count": 50000, "marker": "16S"},
            {"run_accession": "SRR16S_03", "sample_title": "LUCAS_50", "read_count": 50000, "marker": "16S"},
        ])
        df_its = pd.DataFrame([
            {"run_accession": "SRRITS_01", "sample_title": "ITS__50", "read_count": 3000, "marker": "ITS"},
        ])
        paired, metrics = pair_amplicon_runs(df_16s, df_its)
        assert len(paired) == 1
        assert paired["barcode_id"].iloc[0] == 50


# ============================================================================
# 2. Pairing Engine: select_best_run_per_sample Tie-Breaking Probes
# ============================================================================

class TestPairingEngineTieBreaking:
    """Stress-test tie-breaking rules, missing depth attributes, and multi-run selection."""

    def test_tie_breaking_higher_read_depth_wins(self):
        """Candidate run with higher read depth is selected over lower read depth."""
        runs = pd.DataFrame([
            {"run_accession": "SRR002", "sample_title": "LUCAS_77", "read_count": 15000, "marker": "16S"},
            {"run_accession": "SRR001", "sample_title": "LUCAS_77", "read_count": 85000, "marker": "16S"},
        ])
        best = select_best_run_per_sample(runs, "16S")
        assert len(best) == 1
        assert best["run_accession"].iloc[0] == "SRR001"
        assert best["read_count"].iloc[0] == 85000

    def test_tie_breaking_identical_depth_lexicographic_run_accession(self):
        """When read counts are identical, tie is broken by run_accession ASC."""
        runs = pd.DataFrame([
            {"run_accession": "SRR009", "sample_title": "LUCAS_77", "read_count": 50000, "marker": "16S"},
            {"run_accession": "SRR001", "sample_title": "LUCAS_77", "read_count": 50000, "marker": "16S"},
            {"run_accession": "SRR005", "sample_title": "LUCAS_77", "read_count": 50000, "marker": "16S"},
        ])
        best = select_best_run_per_sample(runs, "16S")
        assert len(best) == 1
        assert best["run_accession"].iloc[0] == "SRR001"

    def test_tie_breaking_missing_read_count_column(self):
        """Missing read depth must not be fabricated."""
        runs = pd.DataFrame([
            {"run_accession": "SRR099", "sample_title": "LUCAS_77", "marker": "16S"},
            {"run_accession": "SRR002", "sample_title": "LUCAS_77", "marker": "16S"},
        ])
        with pytest.raises(DataPairingError, match="Missing read depth column"):
            select_best_run_per_sample(runs, "16S")

    def test_tie_breaking_nan_or_corrupt_read_count_coerced_to_zero(self):
        """NaN or non-numeric read count is coerced to 0, preferring valid numerical runs."""
        runs = pd.DataFrame([
            {"run_accession": "SRR_BAD", "sample_title": "LUCAS_77", "read_count": np.nan, "marker": "16S"},
            {"run_accession": "SRR_GOOD", "sample_title": "LUCAS_77", "read_count": 25000, "marker": "16S"},
        ])
        best = select_best_run_per_sample(runs, "16S")
        assert len(best) == 1
        assert best["run_accession"].iloc[0] == "SRR_GOOD"
        assert best["read_count"].iloc[0] == 25000

    def test_all_runs_below_depth_cutoff_fail_qc(self):
        """If all runs for a sample fail minimum read depth, qc_pass is False."""
        runs_16s = pd.DataFrame([
            {"run_accession": "SRR001", "sample_title": "LUCAS_77", "read_count": 5000, "marker": "16S"},  # < 10000
        ])
        runs_its = pd.DataFrame([
            {"run_accession": "SRR002", "sample_title": "ITS__77", "read_count": 3000, "marker": "ITS"},
        ])
        cfg = PairingConfig(min_reads_16s=10000, min_reads_its=500)
        paired, metrics = pair_amplicon_runs(runs_16s, runs_its, config=cfg)
        assert len(paired) == 0  # Filtered out due to failing QC


# ============================================================================
# 3. Pairing Engine: Duplicate Site Deduplication & Canonical Site IDs
# ============================================================================

class TestPairingEngineDuplicateSites:
    """Stress-test canonical site mapping and cross-validation spatial leakage guards."""

    def test_duplicate_pairs_receive_identical_canonical_site_ids(self):
        """Duplicate pairs (59/138, 102/601, 113/717, 116/258) receive identical site_id."""
        df = pd.DataFrame([
            {"barcode_id": 59},
            {"barcode_id": 138},
            {"barcode_id": 102},
            {"barcode_id": 601},
            {"barcode_id": 113},
            {"barcode_id": 717},
            {"barcode_id": 116},
            {"barcode_id": 258},
            {"barcode_id": 999},  # Non-duplicate site
        ])
        clustered = deduplicate_resampled_sites(df, strategy=DedupStrategy.CLUSTER_SITES)

        s_map = dict(zip(clustered["barcode_id"], clustered["site_id"]))
        assert s_map[59] == s_map[138] == "LUCAS_SITE_59_138"
        assert s_map[102] == s_map[601] == "LUCAS_SITE_102_601"
        assert s_map[113] == s_map[717] == "LUCAS_SITE_113_717"
        assert s_map[116] == s_map[258] == "LUCAS_SITE_116_258"
        assert s_map[999] == "LUCAS_SITE_999"

    def test_keep_resampled_strategy_drops_initial_and_retains_replacement(self):
        """KEEP_RESAMPLED drops initial barcodes {59, 102, 113, 116}."""
        df = pd.DataFrame([
            {"barcode_id": 59},
            {"barcode_id": 138},
            {"barcode_id": 102},
            {"barcode_id": 601},
            {"barcode_id": 113},
            {"barcode_id": 717},
            {"barcode_id": 116},
            {"barcode_id": 258},
        ])
        dedup = deduplicate_resampled_sites(df, strategy=DedupStrategy.KEEP_RESAMPLED)

        remaining_barcodes = set(dedup["barcode_id"])
        assert remaining_barcodes == {138, 601, 717, 258}
        assert not remaining_barcodes.intersection(RESAMPLED_INITIAL_BARCODES)

    def test_assert_no_resampled_site_leakage_detects_split_pair(self):
        """Guard catches spatial leakage when a duplicate pair is split across folds."""
        train_barcodes = {59, 10, 20}
        test_barcodes = {138, 30, 40}

        with pytest.raises(ResampledSiteCollisionError, match="SPATIAL LEAKAGE"):
            assert_no_resampled_site_leakage(train_barcodes, test_barcodes)

        # Reverse order
        with pytest.raises(ResampledSiteCollisionError, match="SPATIAL LEAKAGE"):
            assert_no_resampled_site_leakage(test_barcodes, train_barcodes)

    def test_assert_no_resampled_site_leakage_passes_when_clustered(self):
        """Guard passes when duplicate pair is kept together in train fold."""
        train_barcodes = {59, 138, 10, 20}
        test_barcodes = {30, 40}
        # Should not raise
        assert_no_resampled_site_leakage(train_barcodes, test_barcodes)


# ============================================================================
# 4. Sample Manifest: Schema, Coordinates & Texture Validation
# ============================================================================

class TestSampleManifestValidation:
    """Stress-test sample manifest schema validation against corrupt inputs."""

    @pytest.fixture
    def valid_manifest(self) -> pd.DataFrame:
        cfg = MockGeneratorConfig(seed=42, n_samples=10)
        return MockDataGenerator(cfg).generate_manifest()

    def test_rejects_missing_mandatory_columns(self, valid_manifest):
        """Dropping any mandatory column triggers ManifestSchemaValidationError."""
        for col in ["latitude", "longitude", "site_id", "sra_16s_run", "esdac_clay"]:
            corrupt = valid_manifest.drop(columns=[col])
            with pytest.raises(ManifestSchemaValidationError, match="Missing mandatory columns"):
                validate_manifest_schema(corrupt)

    def test_rejects_null_in_mandatory_columns(self, valid_manifest):
        """Null or NaN values in mandatory fields trigger validation error."""
        corrupt = valid_manifest.copy()
        corrupt.loc[0, "latitude"] = np.nan
        with pytest.raises(ManifestSchemaValidationError, match="null values"):
            validate_manifest_schema(corrupt)

    def test_rejects_corrupt_latitude_bounds(self, valid_manifest):
        """Latitudes < -90 or > 90 are rejected."""
        corrupt_high = valid_manifest.copy()
        corrupt_high.loc[0, "latitude"] = 90.001
        with pytest.raises(ManifestSchemaValidationError, match="Latitude out of bounds"):
            validate_manifest_schema(corrupt_high)

        corrupt_low = valid_manifest.copy()
        corrupt_low.loc[0, "latitude"] = -90.001
        with pytest.raises(ManifestSchemaValidationError, match="Latitude out of bounds"):
            validate_manifest_schema(corrupt_low)

    def test_rejects_corrupt_longitude_bounds(self, valid_manifest):
        """Longitudes < -180 or > 180 are rejected."""
        corrupt_high = valid_manifest.copy()
        corrupt_high.loc[0, "longitude"] = 180.001
        with pytest.raises(ManifestSchemaValidationError, match="Longitude out of bounds"):
            validate_manifest_schema(corrupt_high)

        corrupt_low = valid_manifest.copy()
        corrupt_low.loc[0, "longitude"] = -180.001
        with pytest.raises(ManifestSchemaValidationError, match="Longitude out of bounds"):
            validate_manifest_schema(corrupt_low)

    def test_rejects_invalid_soil_texture_sum_exceeding_105(self, valid_manifest):
        """Texture sums (clay + silt + sand) exceeding 105% are rejected."""
        corrupt = valid_manifest.copy()
        corrupt.loc[0, "esdac_clay"] = 50.0
        corrupt.loc[0, "esdac_silt"] = 40.0
        corrupt.loc[0, "esdac_sand"] = 20.0  # Sum = 110.0% > 105.0%
        with pytest.raises(ManifestSchemaValidationError, match="Soil texture sum.*exceeds 105%"):
            validate_manifest_schema(corrupt)

    def test_rejects_biological_ph_bounds(self, valid_manifest):
        """pH values outside [2.0, 11.0] are rejected."""
        corrupt_acid = valid_manifest.copy()
        corrupt_acid.loc[0, "esdac_ph"] = 1.9
        with pytest.raises(ManifestSchemaValidationError, match="esdac_ph out of biological bounds"):
            validate_manifest_schema(corrupt_acid)

        corrupt_base = valid_manifest.copy()
        corrupt_base.loc[0, "esdac_ph"] = 11.1
        with pytest.raises(ManifestSchemaValidationError, match="esdac_ph out of biological bounds"):
            validate_manifest_schema(corrupt_base)

    def test_rejects_negative_organic_carbon_and_nitrogen(self, valid_manifest):
        """Negative values for OC or N are rejected."""
        corrupt_oc = valid_manifest.copy()
        corrupt_oc.loc[0, "esdac_oc"] = -0.5
        with pytest.raises(ManifestSchemaValidationError, match="Negative values found in esdac_oc"):
            validate_manifest_schema(corrupt_oc)

        corrupt_n = valid_manifest.copy()
        corrupt_n.loc[0, "esdac_n"] = -0.1
        with pytest.raises(ManifestSchemaValidationError, match="Negative values found in esdac_n"):
            validate_manifest_schema(corrupt_n)

    def test_rejects_duplicate_barcodes(self, valid_manifest):
        """Duplicate barcode_ids in manifest trigger validation error."""
        corrupt = valid_manifest.copy()
        corrupt.loc[1, "barcode_id"] = corrupt.loc[0, "barcode_id"]
        with pytest.raises(ManifestSchemaValidationError, match="Duplicate barcode_ids"):
            validate_manifest_schema(corrupt)

    def test_rejects_invalid_sra_accession_pattern(self, valid_manifest):
        """Non-conforming SRA run accessions are rejected."""
        corrupt = valid_manifest.copy()
        corrupt.loc[0, "sra_16s_run"] = "INVALID_RUN_ACCESSION"
        with pytest.raises(ManifestSchemaValidationError, match="Invalid sra_16s_run accessions"):
            validate_manifest_schema(corrupt)

    def test_rejects_invalid_land_cover_class(self, valid_manifest):
        """Unknown land cover class triggers validation error."""
        corrupt = valid_manifest.copy()
        corrupt.loc[0, "land_cover"] = "Urban / Industrial"
        with pytest.raises(ManifestSchemaValidationError, match="Invalid land_cover categories"):
            validate_manifest_schema(corrupt)
