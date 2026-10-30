"""
tests/test_data_pairing.py
Comprehensive test suite for data pairing, site deduplication, and manifest generation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.esdac_parser import ESDACLucasParser
from src.data.pairing_engine import (
    CANONICAL_SITE_ID_MAP,
    RESAMPLED_SITE_PAIRS,
    DataPairingError,
    DedupStrategy,
    PairingConfig,
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
    ManifestSchemaValidationError,
    load_paired_manifest,
    save_paired_manifest,
    validate_manifest_schema,
)
from src.data.mock_generator import MockDataGenerator, MockGeneratorConfig


@pytest.fixture
def sample_sra_dfs():
    df_16s = pd.DataFrame([
        {"run_accession": "SRR16S_01", "sample_title": "LUCAS_10", "read_count": 55000, "marker": "16S"},
        {"run_accession": "SRR16S_02", "sample_title": "LUCAS_20", "read_count": 62000, "marker": "16S"},
        {"run_accession": "SRR16S_03", "sample_title": "LUCAS_30", "read_count": 48000, "marker": "16S"},
        {"run_accession": "SRR16S_99", "sample_title": "LUCAS_999", "read_count": 70000, "marker": "16S"},  # Orphan
    ])
    df_its = pd.DataFrame([
        {"run_accession": "SRRITS_01", "sample_title": "ITS__10", "read_count": 3500, "marker": "ITS"},
        {"run_accession": "SRRITS_02", "sample_title": "ITS__20", "read_count": 4200, "marker": "ITS"},
        {"run_accession": "SRRITS_03", "sample_title": "ITS__30", "read_count": 2800, "marker": "ITS"},
        {"run_accession": "SRRITS_88", "sample_title": "ITS__888", "read_count": 5000, "marker": "ITS"},  # Orphan
    ])
    return df_16s, df_its


def test_extract_barcode_id_formats():
    """Verify regex barcode extraction across various string naming conventions."""
    assert extract_barcode_id("LUCAS_92") == 92
    assert extract_barcode_id("lucas-365") == 365
    assert extract_barcode_id("92_HJVG7DRXX") == 92
    assert extract_barcode_id("ITS__92") == 92
    assert extract_barcode_id("ITS_450") == 450
    assert extract_barcode_id("92") == 92
    assert extract_barcode_id(92) == 92
    assert extract_barcode_id("invalid_no_digit") is None


def test_strict_pairing_matching_barcodes(sample_sra_dfs):
    """Verify that only matching barcodes (10, 20, 30) are paired."""
    df_16s, df_its = sample_sra_dfs
    paired_df, metrics = pair_amplicon_runs(df_16s, df_its)

    assert len(paired_df) == 3
    assert set(paired_df["barcode_id"]) == {10, 20, 30}
    assert metrics.validation_status == "PASSED"
    assert (paired_df["sra_16s_run"].str.startswith("SRR16S_")).all()
    assert (paired_df["sra_its_run"].str.startswith("SRRITS_")).all()


def test_orphan_detection_and_exclusion(sample_sra_dfs):
    """Verify orphan runs (999 in 16S, 888 in ITS) are excluded and recorded."""
    df_16s, df_its = sample_sra_dfs
    paired_df, metrics = pair_amplicon_runs(df_16s, df_its)

    assert 999 not in paired_df["barcode_id"].values
    assert 888 not in paired_df["barcode_id"].values
    assert 999 in metrics.orphan_16s_barcodes
    assert 888 in metrics.orphan_its_barcodes


def test_multi_run_tie_breaking():
    """Verify that multiple runs for the same sample are resolved by highest read depth."""
    multi_df = pd.DataFrame([
        {"run_accession": "SRR_LOW", "sample_title": "LUCAS_92", "read_count": 12000, "marker": "16S"},
        {"run_accession": "SRR_HIGH", "sample_title": "LUCAS_92", "read_count": 148000, "marker": "16S"},
    ])
    cfg = PairingConfig(min_reads_16s=10000)
    best = select_best_run_per_sample(multi_df, "16S", cfg)

    assert len(best) == 1
    assert best["run_accession"].iloc[0] == "SRR_HIGH"
    assert best["read_count"].iloc[0] == 148000


def test_anti_synthetic_pairing_assertion():
    """Verify that cross-pairing non-identical biological units raises an error."""
    with pytest.raises(ResampledSiteCollisionError):
        # Assert split duplicate site pair across folds
        assert_no_resampled_site_leakage({59, 100}, {138, 200})


def test_deduplication_drop_initial_sites():
    """Verify that KEEP_RESAMPLED drops {59, 102, 113, 116} and keeps {138, 601, 717, 258}."""
    df = pd.DataFrame([
        {"barcode_id": 59},
        {"barcode_id": 138},
        {"barcode_id": 102},
        {"barcode_id": 601},
        {"barcode_id": 99},
    ])
    dedup = deduplicate_resampled_sites(df, strategy=DedupStrategy.KEEP_RESAMPLED)

    assert 59 not in dedup["barcode_id"].values
    assert 102 not in dedup["barcode_id"].values
    assert 138 in dedup["barcode_id"].values
    assert 601 in dedup["barcode_id"].values
    assert 99 in dedup["barcode_id"].values


def test_canonical_site_id_assignment():
    """Verify that duplicate sites receive unified canonical site IDs."""
    df = pd.DataFrame([
        {"barcode_id": 59},
        {"barcode_id": 138},
        {"barcode_id": 102},
        {"barcode_id": 601},
        {"barcode_id": 99},
    ])
    dedup = deduplicate_resampled_sites(df, strategy=DedupStrategy.CLUSTER_SITES)

    s59 = dedup[dedup["barcode_id"] == 59]["site_id"].iloc[0]
    s138 = dedup[dedup["barcode_id"] == 138]["site_id"].iloc[0]
    assert s59 == s138 == "LUCAS_SITE_59_138"

    s102 = dedup[dedup["barcode_id"] == 102]["site_id"].iloc[0]
    s601 = dedup[dedup["barcode_id"] == 601]["site_id"].iloc[0]
    assert s102 == s601 == "LUCAS_SITE_102_601"


def test_manifest_schema_validation_success():
    """Verify valid synthetic manifest passes schema validation."""
    cfg = MockGeneratorConfig(seed=42, n_samples=20)
    manifest = MockDataGenerator(cfg).generate_manifest()

    audit = validate_manifest_schema(manifest)
    assert audit["status"] == "VALID"
    assert audit["sample_count"] == 20


def test_manifest_schema_out_of_bounds():
    """Verify out-of-bounds parameters trigger ManifestSchemaValidationError."""
    cfg = MockGeneratorConfig(seed=42, n_samples=20)
    manifest = MockDataGenerator(cfg).generate_manifest()

    # Corrupt pH
    bad_manifest = manifest.copy()
    bad_manifest.loc[0, "esdac_ph"] = 14.5
    with pytest.raises(ManifestSchemaValidationError, match="esdac_ph out of biological bounds"):
        validate_manifest_schema(bad_manifest)

    # Corrupt latitude
    bad_lat_manifest = manifest.copy()
    bad_lat_manifest.loc[0, "latitude"] = 95.0
    with pytest.raises(ManifestSchemaValidationError, match="Latitude out of bounds"):
        validate_manifest_schema(bad_lat_manifest)


def test_manifest_csv_round_trip(tmp_path: Path):
    """Verify saving and loading manifest produces bit-for-bit identical DataFrame."""
    cfg = MockGeneratorConfig(seed=42, n_samples=25)
    manifest = MockDataGenerator(cfg).generate_manifest()

    manifest_path = tmp_path / "test_manifest.csv"
    save_paired_manifest(manifest, manifest_path, validate=True)

    loaded = load_paired_manifest(manifest_path, validate=True)
    pd.testing.assert_frame_equal(manifest, loaded)


def test_missing_read_depth_is_not_fabricated():
    """Absent read counts must raise instead of inventing 50,000 / 2,000 reads."""
    runs = pd.DataFrame([
        {"run_accession": "SRR100077", "sample_title": "LUCAS_77", "marker": "16S"},
    ])
    with pytest.raises(DataPairingError, match="Missing read depth column"):
        select_best_run_per_sample(runs, "16S")


def test_enforce_strict_pairing_validates_parsed_esdac_manifest():
    """enforce_strict_pairing(runs_16s, runs_its, esdac_df) must pass schema validation."""
    runs_16s = pd.DataFrame([
        {"run_accession": "SRR100010", "sample_title": "LUCAS_10", "read_count": 45000, "marker": "16S"},
        {"run_accession": "SRR100020", "sample_title": "LUCAS_20", "read_count": 52000, "marker": "16S"},
        {"run_accession": "SRR100030", "sample_title": "LUCAS_30", "read_count": 61000, "marker": "16S"},
    ])
    runs_its = pd.DataFrame([
        {"run_accession": "SRR200010", "sample_title": "ITS__10", "read_count": 2500, "marker": "ITS"},
        {"run_accession": "SRR200020", "sample_title": "ITS__20", "read_count": 3100, "marker": "ITS"},
        {"run_accession": "SRR200030", "sample_title": "ITS__30", "read_count": 2800, "marker": "ITS"},
    ])
    raw_esdac = pd.DataFrame([
        {
            "barcode_id": 10, "gps_lat": 48.12, "gps_long": 2.35, "lc1": "B11",
            "ph_in_h2o": 6.8, "oc": 18.0, "n": 1.6, "clay": 22.0, "silt": 40.0, "sand": 38.0,
        },
        {
            "barcode_id": 20, "gps_lat": 51.50, "gps_long": 7.10, "lc1": "E10",
            "ph_in_h2o": 6.1, "oc": 35.0, "n": 2.8, "clay": 18.0, "silt": 36.0, "sand": 46.0,
        },
        {
            "barcode_id": 30, "gps_lat": 60.20, "gps_long": 17.40, "lc1": "C21",
            "ph_in_h2o": 4.6, "oc": 52.0, "n": 2.4, "clay": 12.0, "silt": 28.0, "sand": 60.0,
        },
    ])
    esdac_df = ESDACLucasParser().parse_dataframe(raw_esdac)
    paired = enforce_strict_pairing(runs_16s, runs_its, esdac_df)

    audit = validate_manifest_schema(paired)
    assert audit["status"] == "VALID"
    assert audit["sample_count"] == 3
    assert set(paired["barcode_id"].astype(int)) == {10, 20, 30}
    assert set(paired["land_cover"]) == {"Cropland", "Grassland", "Woodland"}
    assert paired["land_cover_code"].notna().all()
    pd.testing.assert_series_equal(
        paired["esdac_ph"].reset_index(drop=True),
        paired["ph_h2o"].reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        paired["esdac_oc"].reset_index(drop=True),
        paired["organic_carbon"].reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        paired["land_cover_code"].reset_index(drop=True),
        paired["fine_land_cover_code"].reset_index(drop=True),
        check_names=False,
    )
