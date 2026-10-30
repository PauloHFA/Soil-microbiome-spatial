"""
alpha_diversity.py - Alpha diversity indices: Shannon, Simpson, Chao1, and Observed Richness.
"""

from typing import List
import numpy as np
import pandas as pd


def calculate_observed_richness(counts: np.ndarray) -> int:
    """Number of observed taxa with non-zero abundance."""
    c = np.asarray(counts)
    return int(np.sum(c > 0))


def calculate_shannon(counts: np.ndarray) -> float:
    """Shannon diversity index: H' = -sum(p_i * ln(p_i))."""
    c = np.asarray(counts, dtype=np.float64)
    total = np.sum(c)
    if total <= 0:
        return 0.0
    p = c[c > 0] / total
    return float(-np.sum(p * np.log(p)))


def calculate_simpson(counts: np.ndarray) -> float:
    """Gini-Simpson index: 1 - sum(p_i^2)."""
    c = np.asarray(counts, dtype=np.float64)
    total = np.sum(c)
    if total <= 0:
        return 0.0
    p = c / total
    return float(1.0 - np.sum(p ** 2))


def calculate_chao1(counts: np.ndarray) -> float:
    """
    Chao1 richness estimator:
    S_chao1 = S_obs + (f1 * (f1 - 1)) / (2 * (f2 + 1)) if f2 > 0 else S_obs + (f1 * (f1 - 1)) / 2
    """
    c = np.asarray(counts, dtype=np.int64)
    s_obs = np.sum(c > 0)
    f1 = np.sum(c == 1)
    f2 = np.sum(c == 2)
    if f2 > 0:
        return float(s_obs + (f1 * (f1 - 1)) / (2.0 * (f2 + 1)))
    return float(s_obs + (f1 * (f1 - 1)) / 2.0)


def compute_alpha_diversity_table(
    counts: np.ndarray,
    sample_ids: List[str],
    prefix: str = ""
) -> pd.DataFrame:
    """
    Compute comprehensive alpha diversity metrics for all samples in count matrix.
    """
    pfx = f"{prefix}_" if prefix else ""
    records = []
    for i, sid in enumerate(sample_ids):
        row = counts[i, :]
        records.append({
            "sample_id": sid,
            f"{pfx}observed_richness": calculate_observed_richness(row),
            f"{pfx}shannon": calculate_shannon(row),
            f"{pfx}simpson": calculate_simpson(row),
            f"{pfx}chao1": calculate_chao1(row),
        })
    return pd.DataFrame(records)
