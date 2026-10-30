"""
count_matrix.py - ASV and OTU abundance matrix ingestion, validation, and storage.
"""

from pathlib import Path
from typing import Tuple, List, Union
import numpy as np
import pandas as pd
from scipy import sparse


def load_count_matrix(
    filepath: Union[str, Path]
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load ASV/OTU count table from CSV.
    First column contains sample IDs; header contains ASV/OTU identifiers.

    Returns:
        counts: (n_samples, n_features) non-negative integer ndarray
        sample_ids: list of sample identifiers
        feature_ids: list of ASV/OTU column identifiers
    """
    df = pd.read_csv(filepath, index_col=0)
    sample_ids = [str(s) for s in df.index]
    feature_ids = [str(f) for f in df.columns]
    counts = df.values.astype(np.int64)

    if (counts < 0).any():
        raise ValueError(f"Abundance matrix {filepath} contains negative count values")

    return counts, sample_ids, feature_ids


def save_count_matrix(
    counts: np.ndarray,
    sample_ids: List[str],
    feature_ids: List[str],
    filepath: Union[str, Path]
) -> None:
    """Save count matrix to CSV with sample IDs and feature column names."""
    df = pd.DataFrame(counts, index=sample_ids, columns=feature_ids)
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p)


def to_sparse_matrix(counts: np.ndarray) -> sparse.csr_matrix:
    """Convert dense count array to SciPy CSR sparse matrix."""
    return sparse.csr_matrix(counts)
