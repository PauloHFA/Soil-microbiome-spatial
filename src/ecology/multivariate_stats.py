"""
multivariate_stats.py - Permutational Multivariate Analysis of Variance (PERMANOVA)
and test of homogeneity of multivariate dispersions (betadisper).
"""

from typing import Dict, Any, Union
import numpy as np
from scipy import stats


def _compute_permanova_f(d_sq: np.ndarray, groupings: np.ndarray) -> float:
    """
    Compute Anderson (2001) pseudo-F statistic from squared distance matrix and group labels.
    """
    n = len(groupings)
    unique_groups, group_counts = np.unique(groupings, return_counts=True)
    k = len(unique_groups)

    if k <= 1 or n <= k:
        return 0.0

    ss_total = np.sum(d_sq) / (2.0 * n)

    ss_within = 0.0
    for g, count in zip(unique_groups, group_counts):
        if count > 1:
            mask = groupings == g
            sub_d_sq = d_sq[np.ix_(mask, mask)]
            ss_within += np.sum(sub_d_sq) / (2.0 * count)

    ss_between = ss_total - ss_within
    df_between = k - 1
    df_within = n - k

    if ss_within <= 0 or df_within <= 0:
        return 0.0

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    return float(ms_between / ms_within)


def run_permanova(
    distance_matrix: np.ndarray,
    groupings: Union[list, np.ndarray],
    permutations: int = 999,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run native Anderson (2001) PERMANOVA test on a symmetric distance matrix.

    Args:
        distance_matrix: (N, N) symmetric dissimilarity/distance matrix
        groupings: array of group labels of length N
        permutations: number of Monte Carlo label shuffles
        seed: random seed for reproducibility

    Returns:
        Dict with keys:
            'f_statistic': observed pseudo-F
            'p_value': permutation p-value
            'permutations': number of permutations executed
            'r2': explained variance proportion (SS_between / SS_total)
    """
    d_mat = np.asarray(distance_matrix, dtype=np.float64)
    groups = np.asarray(groupings)
    n = len(groups)

    if d_mat.shape != (n, n):
        raise ValueError(f"Distance matrix shape {d_mat.shape} must match grouping length {n}")

    d_sq = d_mat ** 2
    f_obs = _compute_permanova_f(d_sq, groups)

    ss_total = np.sum(d_sq) / (2.0 * n)
    unique_groups, group_counts = np.unique(groups, return_counts=True)
    ss_within = 0.0
    for g, count in zip(unique_groups, group_counts):
        if count > 1:
            mask = groups == g
            sub_d_sq = d_sq[np.ix_(mask, mask)]
            ss_within += np.sum(sub_d_sq) / (2.0 * count)
    r2 = float(max(0.0, min(1.0, (ss_total - ss_within) / ss_total))) if ss_total > 0 else 0.0

    # Permutations
    rng = np.random.RandomState(seed)
    perm_ge = 0
    perm_f_stats = []

    for _ in range(permutations):
        shuffled_groups = rng.permutation(groups)
        f_perm = _compute_permanova_f(d_sq, shuffled_groups)
        perm_f_stats.append(f_perm)
        if f_perm >= f_obs:
            perm_ge += 1

    p_value = float((perm_ge + 1) / (permutations + 1))

    return {
        "f_statistic": float(f_obs),
        "p_value": p_value,
        "r2": r2,
        "permutations": permutations,
    }


def run_betadisper(
    distance_matrix: np.ndarray,
    groupings: Union[list, np.ndarray]
) -> Dict[str, Any]:
    """
    Test homogeneity of multivariate dispersions across groups (betadisper equivalent).
    Uses principal coordinates embedding / medoid distances and ANOVA on dispersion.
    """
    d_mat = np.asarray(distance_matrix, dtype=np.float64)
    groups = np.asarray(groupings)
    unique_groups = np.unique(groups)

    if len(unique_groups) <= 1:
        return {"f_statistic": 0.0, "p_value": 1.0, "homogeneous": True}

    group_distances = []
    for g in unique_groups:
        mask = groups == g
        sub_d = d_mat[np.ix_(mask, mask)]
        # Distance to group medoid (sample with minimum sum of distances to group members)
        medoid_idx = np.argmin(np.sum(sub_d, axis=1))
        dists_to_medoid = sub_d[medoid_idx, :]
        group_distances.append(dists_to_medoid)

    f_stat, p_val = stats.f_oneway(*group_distances)
    f_stat = float(np.nan_to_num(f_stat, nan=0.0))
    p_val = float(np.nan_to_num(p_val, nan=1.0))

    return {
        "f_statistic": f_stat,
        "p_value": p_val,
        "homogeneous": bool(p_val >= 0.05),
    }
