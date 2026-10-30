"""
Ecology package for alpha diversity, compositional CLR transforms,
beta diversity distances, and multivariate permutational statistics.
"""

from .alpha_diversity import (
    calculate_shannon,
    calculate_simpson,
    calculate_chao1,
    calculate_observed_richness,
    compute_alpha_diversity_table,
)
from .compositional import clr_transform, robust_clr_transform
from .beta_diversity import bray_curtis_distance_matrix, aitchison_distance_matrix
from .multivariate_stats import run_permanova, run_betadisper

__all__ = [
    "calculate_shannon",
    "calculate_simpson",
    "calculate_chao1",
    "calculate_observed_richness",
    "compute_alpha_diversity_table",
    "clr_transform",
    "robust_clr_transform",
    "bray_curtis_distance_matrix",
    "aitchison_distance_matrix",
    "run_permanova",
    "run_betadisper",
]
