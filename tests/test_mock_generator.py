"""
Unit tests for deterministic mock dataset generator (src/data/mock_generator.py).
Asserts exact compliance with Labouyrie et al. (2023) distribution, physical bounds,
reproducibility with fixed seed, and FASTQ sequence integrity.
"""

import gzip
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.mock_generator import (
    EMPIRICAL_CLASS_DISTRIBUTION,
    RESAMPLED_BARCODE_PAIRS,
    MockDataGenerator,
    MockGeneratorConfig,
    generate_mock_data,
)


@pytest.fixture
def temp_mock_dir(tmp_path: Path) -> Path:
    mock_dir = tmp_path / "mock_data"
    mock_dir.mkdir(parents=True, exist_ok=True)
    return mock_dir


def test_mock_generator_deterministic_seed(temp_mock_dir: Path):
    """Assert running mock generator twice with identical seed produces bit-for-bit identical outputs."""
    cfg1 = MockGeneratorConfig(seed=42, n_samples=24, output_dir=str(temp_mock_dir / "run1"))
    cfg2 = MockGeneratorConfig(seed=42, n_samples=24, output_dir=str(temp_mock_dir / "run2"))

    paths1 = MockDataGenerator(cfg1).generate_all()
    paths2 = MockDataGenerator(cfg2).generate_all()

    df_manifest1 = pd.read_csv(paths1["manifest"])
    df_manifest2 = pd.read_csv(paths2["manifest"])
    pd.testing.assert_frame_equal(df_manifest1, df_manifest2)

    df_16s_1 = pd.read_csv(paths1["counts_16s"], index_col=0)
    df_16s_2 = pd.read_csv(paths2["counts_16s"], index_col=0)
    pd.testing.assert_frame_equal(df_16s_1, df_16s_2)

    df_its_1 = pd.read_csv(paths1["counts_its"], index_col=0)
    df_its_2 = pd.read_csv(paths2["counts_its"], index_col=0)
    pd.testing.assert_frame_equal(df_its_1, df_its_2)


def test_mock_generator_715_class_distribution(temp_mock_dir: Path):
    """Assert full cohort generates exactly 715 samples matching Labouyrie et al. 2023 counts."""
    cfg = MockGeneratorConfig(seed=42, n_samples=715, output_dir=str(temp_mock_dir))
    paths = MockDataGenerator(cfg).generate_all()
    df = pd.read_csv(paths["manifest"])

    assert len(df) == 715, f"Expected 715 samples, got {len(df)}"

    # Check exact counts per fine class
    counts = df["fine_land_cover"].value_counts().to_dict()
    for cls, meta in EMPIRICAL_CLASS_DISTRIBUTION.items():
        assert counts.get(cls) == meta["count"], (
            f"Class '{cls}' count mismatch: expected {meta['count']}, got {counts.get(cls)}"
        )

    # Check macro categories
    macro_counts = df["macro_land_cover"].value_counts().to_dict()
    assert macro_counts["Cropland"] == 310
    assert macro_counts["Grassland"] == 146
    assert macro_counts["Woodland"] == 259


def test_mock_generator_strict_pairing_integrity(temp_mock_dir: Path):
    """Assert all generated records have both valid 16S and ITS SRA runs and non-null sample IDs."""
    cfg = MockGeneratorConfig(seed=42, n_samples=50, output_dir=str(temp_mock_dir))
    paths = MockDataGenerator(cfg).generate_all()
    df = pd.read_csv(paths["manifest"])

    assert df["sample_id"].notna().all()
    assert df["barcode_id"].notna().all()
    assert (df["barcode_id"] > 0).all()
    assert df["sra_16s_run"].notna().all()
    assert df["sra_its_run"].notna().all()
    assert (df["reads_16s"] > 1000).all()
    assert (df["reads_its"] > 100).all()
    assert (df["include_in_core"] == True).all()


def test_mock_generator_edaphic_physical_bounds(temp_mock_dir: Path):
    """Assert physicochemical and spatial bounds adhere to realistic soil chemistry and geography."""
    cfg = MockGeneratorConfig(seed=42, n_samples=100, output_dir=str(temp_mock_dir))
    paths = MockDataGenerator(cfg).generate_all()
    df = pd.read_csv(paths["manifest"])

    # pH bounds
    assert df["esdac_ph"].between(3.0, 9.5).all(), "pH out of physical bounds [3.0, 9.5]"
    assert df["ph_in_cacl2"].between(2.5, 9.0).all()
    assert (df["esdac_ph"] >= df["ph_in_cacl2"]).all(), "pH in H2O should be >= pH in CaCl2"

    # Soil texture sums to 100%
    texture_sum = df["esdac_clay"] + df["esdac_silt"] + df["esdac_sand"]
    np.testing.assert_allclose(texture_sum, 100.0, atol=0.1, err_msg="Texture clay+silt+sand != 100%")

    # Organic carbon and nitrogen stoichiometry
    assert (df["esdac_oc"] > 0).all()
    assert (df["esdac_n"] > 0).all()
    computed_cn = df["esdac_oc"] / df["esdac_n"]
    np.testing.assert_allclose(computed_cn, df["c_n_ratio"], rtol=0.05)

    # European spatial bounding box
    assert df["latitude"].between(34.0, 71.0).all(), "Latitude outside Europe"
    assert df["longitude"].between(-11.0, 33.0).all(), "Longitude outside Europe"


def test_mock_generator_asv_matrix_characteristics(temp_mock_dir: Path):
    """Assert ASV abundance matrices have correct shapes, non-negative integers, and realistic sparsity."""
    cfg = MockGeneratorConfig(seed=42, n_samples=30, n_asvs_16s=100, n_asvs_its=80, output_dir=str(temp_mock_dir))
    paths = MockDataGenerator(cfg).generate_all()

    df_16s = pd.read_csv(paths["counts_16s"], index_col=0)
    df_its = pd.read_csv(paths["counts_its"], index_col=0)

    assert df_16s.shape == (30, 100)
    assert df_its.shape == (30, 80)
    assert (df_16s.values >= 0).all()
    assert (df_its.values >= 0).all()

    # Sparsity should be realistic (between 20% and 90% zeros)
    sparsity_16s = (df_16s.values == 0).mean()
    sparsity_its = (df_its.values == 0).mean()
    assert 0.20 <= sparsity_16s <= 0.90, f"Unrealistic 16S sparsity: {sparsity_16s:.2f}"
    assert 0.20 <= sparsity_its <= 0.90, f"Unrealistic ITS sparsity: {sparsity_its:.2f}"


def _assert_resampled_pairs_colocated(df: pd.DataFrame) -> None:
    """Every re-sampled pair present in df must share coordinates and canonical site_id."""
    present = set(df["barcode_id"].astype(int))
    for initial_id, resampled_id in RESAMPLED_BARCODE_PAIRS.items():
        if initial_id not in present or resampled_id not in present:
            continue
        left = df.loc[df["barcode_id"] == initial_id].iloc[0]
        right = df.loc[df["barcode_id"] == resampled_id].iloc[0]
        assert left["latitude"] == right["latitude"]
        assert left["longitude"] == right["longitude"]
        assert left["site_id"] == right["site_id"] == f"LUCAS_SITE_{initial_id}_{resampled_id}"
        if "spatial_group" in df.columns:
            assert left["spatial_group"] == right["spatial_group"]


def test_mock_generator_duplicate_site_resolution(temp_mock_dir: Path):
    """Assert the 715 cohort includes barcode 717 and binds all four re-sampled pairs."""
    cfg = MockGeneratorConfig(seed=42, n_samples=715, output_dir=str(temp_mock_dir))
    df = MockDataGenerator(cfg).generate_manifest()

    assert len(df) == 715
    assert 717 in set(df["barcode_id"].astype(int))
    assert df["land_cover_code"].between(0, 5).all()
    assert (df["fine_land_cover"] == "Intensive grassland").any()
    for initial_id, resampled_id in RESAMPLED_BARCODE_PAIRS.items():
        assert initial_id in set(df["barcode_id"].astype(int))
        assert resampled_id in set(df["barcode_id"].astype(int))
    _assert_resampled_pairs_colocated(df)

    paths = MockDataGenerator(cfg).generate_all()
    written = pd.read_csv(paths["manifest"])
    _assert_resampled_pairs_colocated(written)


def test_mock_generator_colocates_pairs_whenever_both_barcodes_exist():
    """Co-location must not depend on the unreachable len(df) >= 717 guard."""
    df = MockDataGenerator(MockGeneratorConfig(seed=7, n_samples=140)).generate_manifest()
    present = set(df["barcode_id"].astype(int))
    assert {59, 138}.issubset(present)
    assert 717 not in present
    _assert_resampled_pairs_colocated(df)


def test_mock_generator_fastqs_and_primers(temp_mock_dir: Path):
    """Assert generated FASTQs contain expected primers and valid Illumina/PacBio formatting."""
    cfg = MockGeneratorConfig(
        seed=42,
        n_samples=4,
        output_dir=str(temp_mock_dir),
        generate_fastqs=True,
        fastq_sample_limit=2,
        reads_per_fastq=20
    )
    paths = MockDataGenerator(cfg).generate_all()
    fastq_dir = Path(paths["fastqs_dir"])

    r1_files = list(fastq_dir.glob("*_1.fastq.gz"))
    its_files = list(fastq_dir.glob("SRR*407*.fastq.gz"))
    assert len(r1_files) == 2
    assert len(its_files) == 2

    # Verify R1 format and primers
    with gzip.open(r1_files[0], "rt") as f:
        lines = [f.readline().strip() for _ in range(8)]
        assert lines[0].startswith("@SRR")
        assert len(lines[1]) > 100
        assert lines[2] == "+"
        assert len(lines[3]) == len(lines[1])
        assert ("GTGYCAGCMGCCGCGGTAA" in lines[1]) or ("GGCCGYCAATTYMTTTRAGTTT" in lines[1])
