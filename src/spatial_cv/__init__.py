"""
Spatial cross-validation package: spatial blocking, buffering, and leakage guards.
"""

from .spatial_partitioner import (
    SpatialSplitPlan,
    SpatialValidationError,
    apply_spatial_buffering,
    create_spatial_splits,
    create_validated_spatial_splits,
    haversine_distance_matrix,
    test_categories_represented,
)
from .leakage_guard import assert_zero_spatial_leakage, assert_no_duplicate_site_split

__all__ = [
    "SpatialSplitPlan",
    "SpatialValidationError",
    "apply_spatial_buffering",
    "create_spatial_splits",
    "create_validated_spatial_splits",
    "haversine_distance_matrix",
    "test_categories_represented",
    "assert_zero_spatial_leakage",
    "assert_no_duplicate_site_split",
]
