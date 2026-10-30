"""
normalization.py - Sequencing depth filtering and Scaling with Ranked Subsampling (SRS).
"""

from typing import Tuple, List, Optional
import numpy as np


def filter_by_depth(
    counts: np.ndarray,
    sample_ids: List[str],
    min_depth: int
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Filter out samples with total sequencing read depth below min_depth.

    Returns:
        filtered_counts, filtered_sample_ids, retained_mask
    """
    depths = np.sum(counts, axis=1)
    mask = depths >= min_depth
    return counts[mask], [sid for sid, m in zip(sample_ids, mask) if m], mask


def normalize_sample_srs(
    counts: np.ndarray,
    target_depth: int,
    seed: int = 42
) -> np.ndarray:
    """
    Scaling with Ranked Subsampling (SRS) for a single sample vector.
    Conforms to Beule & Karlovsky (2020) SRS algorithm.

    Args:
        counts: 1D array of non-negative integer abundances
        target_depth: target normalization read depth
        seed: random seed for deterministic tie breaking

    Returns:
        1D array of SRS normalized integer counts summing exactly to target_depth
    """
    c = np.array(counts, dtype=np.float64)
    total = np.sum(c)
    if total < target_depth:
        raise ValueError(f"Sample depth {total} is below target normalization depth {target_depth}")

    scaled = (c / total) * target_depth
    floored = np.floor(scaled).astype(np.int64)
    remainder = int(target_depth - np.sum(floored))

    if remainder > 0:
        decimals = scaled - floored
        rng = np.random.RandomState(seed)
        jittered = decimals + rng.uniform(0, 1e-9, size=len(decimals))
        top_indices = np.argsort(jittered)[::-1][:remainder]
        floored[top_indices] += 1

    return floored


def normalize_srs(
    counts: np.ndarray,
    target_depth: int,
    seed: int = 42
) -> np.ndarray:
    """
    Apply Scaling with Ranked Subsampling (SRS) across all samples (rows) in matrix.
    """
    out = np.zeros_like(counts, dtype=np.int64)
    for i in range(counts.shape[0]):
        out[i, :] = normalize_sample_srs(counts[i, :], target_depth, seed=seed + i)
    return out
