"""
compositional.py - Centered Log-Ratio (CLR) and robust CLR transformations.
"""

import numpy as np


def clr_transform(counts: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
    """
    Centered Log-Ratio (CLR) transformation for compositional simplex data.
    x_clr_i = ln(x_i / g(x)) where g(x) is the geometric mean.
    Sum of values across features is identically 0.

    Args:
        counts: 1D or 2D array of abundances
        pseudocount: value to replace zero counts

    Returns:
        CLR-transformed ndarray of same shape
    """
    matrix = np.array(counts, dtype=np.float64)
    matrix_pseudo = np.where(matrix <= 0, pseudocount, matrix)
    log_matrix = np.log(matrix_pseudo)
    geom_mean = np.mean(log_matrix, axis=-1, keepdims=True)
    return log_matrix - geom_mean


def robust_clr_transform(counts: np.ndarray) -> np.ndarray:
    """
    Robust Centered Log-Ratio (rCLR) transformation (Martino et al. 2019).
    Computes geometric mean over observed non-zero values only, leaving true zeros as zero.
    """
    matrix = np.array(counts, dtype=np.float64)
    out = np.zeros_like(matrix)

    if matrix.ndim == 1:
        mask = matrix > 0
        if np.any(mask):
            gm = np.mean(np.log(matrix[mask]))
            out[mask] = np.log(matrix[mask]) - gm
        return out

    for i in range(matrix.shape[0]):
        row = matrix[i, :]
        mask = row > 0
        if np.any(mask):
            gm = np.mean(np.log(row[mask]))
            out[i, mask] = np.log(row[mask]) - gm
    return out
