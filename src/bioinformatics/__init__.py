"""
Bioinformatics package for amplicon QC, matrix ingestion, taxonomy, and normalization.
"""

from .fastq_qc import detect_orientation, reorient_read, trim_primer
from .count_matrix import load_count_matrix, save_count_matrix
from .taxonomy import parse_taxonomy_line, rollup_taxonomy
from .normalization import filter_by_depth, normalize_srs

__all__ = [
    "detect_orientation",
    "reorient_read",
    "trim_primer",
    "load_count_matrix",
    "save_count_matrix",
    "parse_taxonomy_line",
    "rollup_taxonomy",
    "filter_by_depth",
    "normalize_srs",
]
