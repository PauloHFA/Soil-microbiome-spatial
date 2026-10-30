"""
leakage_guard.py - Programmatic assertions and validation for zero spatial data leakage.
"""

from typing import List, Tuple, Set, Any, Optional
import numpy as np
import pandas as pd


def assert_zero_spatial_leakage(
    train_sites: Set[Any],
    test_sites: Set[Any]
) -> None:
    """
    Assert that training and test site partitions are strictly disjoint:
    Sites(train) ∩ Sites(test) = ∅.
    """
    overlap = train_sites.intersection(test_sites)
    if len(overlap) > 0:
        raise AssertionError(
            f"Spatial data leakage detected! {len(overlap)} sites present in both "
            f"train and test sets: {sorted(list(overlap))[:5]}"
        )


def assert_no_duplicate_site_split(
    manifest: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    duplicate_pairs: List[Tuple[int, int]]
) -> None:
    """
    Assert that re-sampled duplicate site pairs never have one sample in train
    and the other in test.
    """
    train_barcodes = set(manifest.iloc[train_idx]["barcode_id"].values)
    test_barcodes = set(manifest.iloc[test_idx]["barcode_id"].values)

    for b1, b2 in duplicate_pairs:
        in_train_1, in_test_1 = b1 in train_barcodes, b1 in test_barcodes
        in_train_2, in_test_2 = b2 in train_barcodes, b2 in test_barcodes

        if (in_train_1 and in_test_2) or (in_train_2 and in_test_1):
            raise AssertionError(
                f"Resampled duplicate site pair ({b1}, {b2}) split across train and test folds! "
                f"Barcode {b1} in train={in_train_1}/test={in_test_1}; "
                f"Barcode {b2} in train={in_train_2}/test={in_test_2}."
            )


def verify_all_splits(
    manifest: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    duplicate_pairs: Optional[List[Tuple[int, int]]] = None
) -> bool:
    """
    Run full battery of zero-leakage assertions across all CV folds.
    """
    site_col = "site_id" if "site_id" in manifest.columns else "sample_id"
    for fold_idx, (tr_idx, te_idx) in enumerate(splits):
        tr_sites = set(manifest.iloc[tr_idx][site_col].values)
        te_sites = set(manifest.iloc[te_idx][site_col].values)
        assert_zero_spatial_leakage(tr_sites, te_sites)

        if duplicate_pairs:
            assert_no_duplicate_site_split(manifest, tr_idx, te_idx, duplicate_pairs)
    return True
