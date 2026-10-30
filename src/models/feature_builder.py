"""
feature_builder.py - Feature extraction, inside-fold scaling, and feature set generation.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from src.ecology.compositional import clr_transform


def build_feature_sets(
    counts_16s: np.ndarray,
    counts_its: np.ndarray,
    pseudocount: float = 1e-6
) -> Dict[str, np.ndarray]:
    """
    Build canonical feature sets A, B, and C for comparative classification:
        - Set A: 16S Bacterial CLR profiles
        - Set B: ITS Fungal CLR profiles
        - Set C: Integrated Multi-marker (Early Fusion: [Set A, Set B])

    Returns:
        Dict with keys: 'set_a_16s', 'set_b_its', 'set_c_integrated'
    """
    clr_16s = clr_transform(counts_16s, pseudocount=pseudocount)
    clr_its = clr_transform(counts_its, pseudocount=pseudocount)
    clr_integrated = np.hstack([clr_16s, clr_its])

    return {
        "set_a_16s": clr_16s,
        "set_b_its": clr_its,
        "set_c_integrated": clr_integrated,
    }


def impute_and_scale_inside_fold(
    x_train: np.ndarray,
    x_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit median imputation and standard scaling strictly on training fold
    and transform both train and test without data leakage.
    """
    # Compute column medians on train (ignoring NaNs)
    col_medians = np.nanmedian(x_train, axis=0)
    col_medians = np.nan_to_num(col_medians, nan=0.0)

    # Impute train
    inds_tr = np.where(np.isnan(x_train))
    x_tr_imp = x_train.copy()
    x_tr_imp[inds_tr] = np.take(col_medians, inds_tr[1])

    # Impute test
    inds_te = np.where(np.isnan(x_test))
    x_te_imp = x_test.copy()
    x_te_imp[inds_te] = np.take(col_medians, inds_te[1])

    # Fit standardization on train only
    mean_tr = np.mean(x_tr_imp, axis=0)
    std_tr = np.std(x_tr_imp, axis=0)
    std_tr[std_tr == 0] = 1.0

    x_tr_scaled = (x_tr_imp - mean_tr) / std_tr
    x_te_scaled = (x_te_imp - mean_tr) / std_tr

    return x_tr_scaled, x_te_scaled
