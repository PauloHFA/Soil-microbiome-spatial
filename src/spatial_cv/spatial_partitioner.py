"""
spatial_partitioner.py - Spatial blocking and buffer exclusion.

Random splits are not a fallback. A test category must already be present
on other training sites. Buffer points are not put back to save a class.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


EARTH_RADIUS_KM = 6371.0088


class SpatialValidationError(Exception):
    """Raised when a split would not be a spatial validation under the revised spec."""


@dataclass(frozen=True)
class SpatialSplitPlan:
    splits: List[Tuple[np.ndarray, np.ndarray]]
    n_splits_requested: int
    n_splits_used: int
    exploratory: bool
    dropped_folds: int


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers. Radius = 6,371.0088 km."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def haversine_distance_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Pairwise Haversine distance matrix (N x N) in km."""
    n = len(lats)
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            distance = haversine_distance(lats[i], lons[i], lats[j], lons[j])
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance
    return dist_matrix


def test_categories_represented(train_labels: Sequence, test_labels: Sequence) -> bool:
    """Every category evaluated in test must already occur on the training side."""
    test_classes = set(np.unique(np.asarray(test_labels)))
    train_classes = set(np.unique(np.asarray(train_labels)))
    return test_classes.issubset(train_classes)


def create_validated_spatial_splits(
    manifest: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    group_col: str = "site_id",
    label_col: Optional[str] = None,
) -> SpatialSplitPlan:
    """
    Group sites into spatial folds.

    If there are fewer sites than requested folds, the split shrinks to
    leave-one-site-out and is marked exploratory. It never becomes a random KFold.
    Folds whose test categories are absent from training are dropped, not repaired
    by moving a site across the split.
    """
    del seed  # Group assignment is deterministic. The seed is recorded by the caller.
    if n_splits < 2:
        raise SpatialValidationError("Spatial validation needs at least two folds.")
    if manifest is None or manifest.empty:
        raise SpatialValidationError("Cannot validate an empty manifest.")
    if group_col not in manifest.columns:
        raise SpatialValidationError(
            f"Column '{group_col}' is required. A random split is not a spatial validation."
        )

    groups = manifest[group_col].astype(str)
    if groups.isna().any() or (groups.str.strip() == "").any() or (groups == "nan").any():
        raise SpatialValidationError("Blank site ids cannot be spatially blocked.")

    n_groups = int(groups.nunique())
    if n_groups < 2:
        raise SpatialValidationError(
            "Fewer than two spatial units. Refusing a random split, and not claiming "
            "geographic validation."
        )

    exploratory = n_groups < n_splits
    n_used = n_groups if exploratory else n_splits
    splitter = GroupKFold(n_splits=n_used)
    raw_splits = list(splitter.split(np.arange(len(manifest)), groups=groups.to_numpy()))

    labels = None
    if label_col is not None:
        if label_col not in manifest.columns:
            raise SpatialValidationError(f"Label column '{label_col}' is missing.")
        labels = manifest[label_col].to_numpy()
        if pd.isna(labels).any():
            raise SpatialValidationError(f"Label column '{label_col}' contains missing values.")

    kept: List[Tuple[np.ndarray, np.ndarray]] = []
    dropped = 0
    for train_idx, test_idx in raw_splits:
        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)
        train_sites = set(groups.iloc[train_idx])
        test_sites = set(groups.iloc[test_idx])
        overlap = train_sites.intersection(test_sites)
        if overlap:
            raise SpatialValidationError(
                f"Same site on both sides of a split: {sorted(overlap)[:5]}"
            )
        if labels is not None and not test_categories_represented(labels[train_idx], labels[test_idx]):
            dropped += 1
            continue
        kept.append((train_idx, test_idx))

    if not kept:
        raise SpatialValidationError(
            "No spatial fold kept every test category in training. "
            "This is not repaired by a random split or by moving sites."
        )
    if dropped:
        exploratory = True

    return SpatialSplitPlan(
        splits=kept,
        n_splits_requested=n_splits,
        n_splits_used=n_used,
        exploratory=exploratory,
        dropped_folds=dropped,
    )


def create_spatial_splits(
    manifest: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    group_col: str = "site_id",
    label_col: Optional[str] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return spatially blocked folds. Raises instead of falling back to KFold."""
    plan = create_validated_spatial_splits(
        manifest,
        n_splits=n_splits,
        seed=seed,
        group_col=group_col,
        label_col=label_col,
    )
    return plan.splits


def apply_spatial_buffering(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    labels: np.ndarray,
    buffer_km: float = 20.0,
) -> np.ndarray:
    """
    Drop training points that fall inside buffer_km of a test point.

    A point inside the buffer is not put back, even when it is the last
    training example of a class. The caller must drop the fold if a test
    category disappears.
    """
    del labels  # Class rescue is intentionally unsupported.
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    if buffer_km <= 0 or len(test_idx) == 0 or len(train_idx) == 0:
        return train_idx

    test_lats = np.asarray(lats)[test_idx]
    test_lons = np.asarray(lons)[test_idx]
    retained: List[int] = []
    for train_pos in train_idx:
        lat_tr = float(lats[train_pos])
        lon_tr = float(lons[train_pos])
        min_dist = min(
            haversine_distance(lat_tr, lon_tr, float(tlat), float(tlon))
            for tlat, tlon in zip(test_lats, test_lons)
        )
        if min_dist >= buffer_km:
            retained.append(int(train_pos))
    return np.asarray(sorted(set(retained)), dtype=np.int64)
