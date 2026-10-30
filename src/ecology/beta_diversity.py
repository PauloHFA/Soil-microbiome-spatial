"""
beta_diversity.py - Pairwise beta diversity dissimilarity and distance matrices.
"""

from typing import Union
import numpy as np
from scipy.spatial.distance import pdist, squareform
from .compositional import clr_transform


def bray_curtis_distance_matrix(counts: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Bray-Curtis dissimilarity matrix across all samples (rows).
    BC(u, v) = sum(|u_i - v_i|) / sum(u_i + v_i)

    Returns:
        (n_samples, n_samples) symmetric distance matrix
    """
    matrix = np.asarray(counts, dtype=np.float64)
    if matrix.shape[0] <= 1:
        return np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.float64)

    condensed = pdist(matrix, metric="braycurtis")
    # Replace NaN (all-zero comparisons) with 0.0
    condensed = np.nan_to_num(condensed, nan=0.0)
    return squareform(condensed)


def aitchison_distance_matrix(
    counts: np.ndarray,
    pseudocount: float = 1e-6
) -> np.ndarray:
    """
    Compute pairwise Aitchison distance matrix (Euclidean distance on CLR vectors).
    d_Aitchison(u, v) = ||clr(u) - clr(v)||_2

    Returns:
        (n_samples, n_samples) symmetric distance matrix
    """
    matrix = np.asarray(counts, dtype=np.float64)
    if matrix.shape[0] <= 1:
        return np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.float64)

    clr_mat = clr_transform(matrix, pseudocount=pseudocount)
    condensed = pdist(clr_mat, metric="euclidean")
    return squareform(condensed)
