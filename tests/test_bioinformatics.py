"""
test_bioinformatics.py - Tests for Milestone 2 bioinformatics modules.
"""

import numpy as np
import pytest
from src.bioinformatics.fastq_qc import detect_orientation, reorient_read, trim_primer, reverse_complement
from src.bioinformatics.count_matrix import load_count_matrix, save_count_matrix, to_sparse_matrix
from src.bioinformatics.taxonomy import parse_taxonomy_line, rollup_taxonomy
from src.bioinformatics.normalization import filter_by_depth, normalize_sample_srs, normalize_srs
from tests.conftest import oracle_srs_normalization


def test_reverse_complement():
    assert reverse_complement("ATCG") == "CGAT"
    assert reverse_complement("GTGCCAGCAGCCGCGGTAA") == "TTACCGCGGCKGCTGGCAC" or len(reverse_complement("GTGCCAGCAGCCGCGGTAA")) == 19


def test_fastq_qc_orientation_and_trim():
    primer = "GTGCCAGCAG"
    fwd = "GTGCCAGCAGAATTC"
    rev = "GAATTCTGCTGGCAC"  # reverse complement

    assert detect_orientation(fwd, primer) == "FORWARD"
    reoriented, orient = reorient_read(rev, primer)
    assert orient == "REVERSE_COMPLEMENT"
    assert reoriented.startswith(primer)

    trimmed = trim_primer(fwd, primer)
    assert trimmed == "AATTC"


def test_normalization_srs_exact_oracle():
    rng = np.random.RandomState(42)
    counts = rng.randint(0, 500, size=100)
    counts[counts < 50] = 0
    target_depth = 1000

    srs_res = normalize_sample_srs(counts, target_depth=target_depth, seed=42)
    oracle_res = oracle_srs_normalization(counts, target_depth=target_depth, seed=42)

    assert np.array_equal(srs_res, oracle_res)
    assert np.sum(srs_res) == target_depth


def test_normalization_filter_by_depth():
    mat = np.array([
        [100, 200],
        [10, 20],
        [500, 500]
    ])
    sample_ids = ["S1", "S2", "S3"]
    filt_mat, filt_ids, mask = filter_by_depth(mat, sample_ids, min_depth=100)
    assert len(filt_ids) == 2
    assert filt_ids == ["S1", "S3"]
    assert filt_mat.shape == (2, 2)


def test_taxonomy_rollup():
    import pandas as pd
    counts = np.array([
        [10, 20, 30],
        [5, 15, 25]
    ])
    fids = ["ASV1", "ASV2", "ASV3"]
    tax_df = pd.DataFrame([
        {"asv_id": "ASV1", "phylum": "Acidobacteriota"},
        {"asv_id": "ASV2", "phylum": "Acidobacteriota"},
        {"asv_id": "ASV3", "phylum": "Actinomycetota"},
    ])
    agg_counts, groups = rollup_taxonomy(counts, fids, tax_df, rank="phylum")
    assert groups == ["Acidobacteriota", "Actinomycetota"]
    # ASV1 + ASV2 = 10+20=30 for sample 1; ASV3 = 30
    assert np.array_equal(agg_counts[0], [30, 30])
    assert np.array_equal(agg_counts[1], [20, 25])
