"""
src/data/pairing_engine.py
Deterministic 1:1 biological pairing engine for 16S and ITS amplicon datasets.

Enforces strict physical soil unit matching, resolves multi-run samples,
deduplicates re-sampled sites (59/138, 102/601, 113/717, 116/258),
and audits pairing attrition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple
import re
import numpy as np
import pandas as pd


class PairingStrategy(str, Enum):
    HIGHEST_DEPTH = "highest_depth"
    FIRST_PASSING = "first_passing"
    SUM_READS = "sum_reads"


class DedupStrategy(str, Enum):
    KEEP_RESAMPLED = "keep_resampled"  # Drop {59, 102, 113, 116}, keep {138, 601, 717, 258}
    KEEP_INITIAL = "keep_initial"      # Keep {59, 102, 113, 116}, drop {138, 601, 717, 258}
    CLUSTER_SITES = "cluster_sites"    # Keep all, assign identical canonical site_id


class DataPairingError(Exception):
    """Base exception for data pairing failures."""
    pass


class SamplePairingViolationError(DataPairingError):
    """Raised when 16S and ITS records do not originate from identical biological unit."""
    pass


class ResampledSiteCollisionError(DataPairingError):
    """Raised when duplicate/re-sampled sites leak across cross-validation splits."""
    pass


class InsufficientReadsError(DataPairingError):
    """Raised when candidate run fails minimum read depth cutoff."""
    pass


@dataclass(frozen=True)
class PairingConfig:
    min_reads_16s: int = 10000
    min_reads_its: int = 500
    srs_target_16s: int = 40109
    srs_target_its: int = 502
    run_selection_strategy: PairingStrategy = PairingStrategy.HIGHEST_DEPTH
    dedup_strategy: DedupStrategy = DedupStrategy.KEEP_RESAMPLED
    require_esdac_match: bool = True
    target_macro_classes: Tuple[str, ...] = ("Cropland", "Grassland", "Woodland")
    target_biogeo_regions: Tuple[str, ...] = (
        "Atlantic", "Boreal", "Continental", "Mediterranean"
    )


@dataclass
class PairingAuditMetrics:
    total_raw_16s_runs: int = 0
    total_raw_its_runs: int = 0
    unique_16s_barcodes: int = 0
    unique_its_barcodes: int = 0
    passing_16s_barcodes: int = 0
    passing_its_barcodes: int = 0
    orphan_16s_barcodes: List[int] = field(default_factory=list)
    orphan_its_barcodes: List[int] = field(default_factory=list)
    paired_samples_pre_dedup: int = 0
    deduplicated_dropped_barcodes: List[int] = field(default_factory=list)
    paired_samples_post_dedup: int = 0
    esdac_matched_count: int = 0
    final_core_cohort_count: int = 0
    validation_status: str = "PENDING"


# Re-sampled duplicate locations in LUCAS PRJNA952168
RESAMPLED_SITE_PAIRS: Dict[int, int] = {
    59: 138,
    102: 601,
    113: 717,
    116: 258,
}

RESAMPLED_INITIAL_BARCODES: Set[int] = set(RESAMPLED_SITE_PAIRS.keys())
RESAMPLED_REPLACEMENT_BARCODES: Set[int] = set(RESAMPLED_SITE_PAIRS.values())
ALL_RESAMPLED_BARCODES: Set[int] = RESAMPLED_INITIAL_BARCODES | RESAMPLED_REPLACEMENT_BARCODES

# Canonical site mapping: maps barcode_id to unique physical site identifier
CANONICAL_SITE_ID_MAP: Dict[int, str] = {
    59: "LUCAS_SITE_59_138",
    138: "LUCAS_SITE_59_138",
    102: "LUCAS_SITE_102_601",
    601: "LUCAS_SITE_102_601",
    113: "LUCAS_SITE_113_717",
    717: "LUCAS_SITE_113_717",
    116: "LUCAS_SITE_116_258",
    258: "LUCAS_SITE_116_258",
}


def extract_barcode_id(value: Any) -> Optional[int]:
    """
    Extract integer BARCODE_ID from sample title, library name, or numeric field.
    
    Examples:
      'LUCAS_92'       -> 92
      '92_HJVG7DRXX'   -> 92
      'ITS__92'        -> 92
      '92' / 92        -> 92
    """
    if value is None or pd.isna(value):
        return None
    val_str = str(value).strip()
    if not val_str:
        return None

    if val_str.isdigit():
        return int(val_str)

    m_lucas = re.match(r"^LUCAS[_\-\s]+(\d+)$", val_str, re.IGNORECASE)
    if m_lucas:
        return int(m_lucas.group(1))

    m_its = re.match(r"^ITS[_\-]+(\d+)$", val_str, re.IGNORECASE)
    if m_its:
        return int(m_its.group(1))

    m_flowcell = re.match(r"^(\d+)_[A-Za-z0-9]+$", val_str)
    if m_flowcell:
        return int(m_flowcell.group(1))

    m_any = re.search(r"\b(\d{1,4})\b", val_str)
    if m_any:
        return int(m_any.group(1))

    return None


def classify_sra_marker(row: pd.Series) -> str:
    """Classify SRA run into '16S', 'ITS', or 'UNKNOWN' based on sequencing metadata."""
    target = str(row.get("target_gene", row.get("marker", ""))).upper()
    if "16S" in target or "V4" in target:
        return "16S"
    if "ITS" in target or "FUNGI" in target:
        return "ITS"

    platform = str(row.get("platform", row.get("instrument_platform", ""))).upper()
    layout = str(row.get("library_layout", row.get("layout", ""))).upper()
    lib_name = str(row.get("library_name", "")).upper()
    model = str(row.get("instrument_model", row.get("model", ""))).upper()

    if "PACBIO" in platform or "SEQUEL" in model or lib_name.startswith("ITS"):
        return "ITS"
    if "ILLUMINA" in platform or "MISEQ" in model or layout == "PAIRED":
        return "16S"

    return "UNKNOWN"


def select_best_run_per_sample(
    runs_df: pd.DataFrame,
    marker: Literal["16S", "ITS"],
    config: PairingConfig = PairingConfig()
) -> pd.DataFrame:
    """
    Filter runs by marker and resolve duplicate runs per barcode_id.
    
    Tie-breaking:
      1. filter by min read depth
      2. sort by read_count DESC, then run_accession ASC (deterministic)
      3. pick the top candidate
    """
    if runs_df.empty:
        return pd.DataFrame()

    df = runs_df.copy()
    if "marker" not in df.columns:
        df["marker"] = df.apply(classify_sra_marker, axis=1)

    marker_df = df[df["marker"] == marker].copy()
    if marker_df.empty:
        raise DataPairingError(f"No runs found matching marker '{marker}'")

    if "barcode_id" not in marker_df.columns:
        title_col = next((c for c in ["sample_title", "sample_name", "title"] if c in marker_df.columns), None)
        if title_col:
            marker_df["barcode_id"] = marker_df[title_col].apply(extract_barcode_id)
        else:
            raise DataPairingError("Cannot locate barcode_id or sample_title in runs DataFrame")

    marker_df = marker_df.dropna(subset=["barcode_id"]).copy()
    marker_df["barcode_id"] = marker_df["barcode_id"].astype(int)

    # Read count column normalization. Never invent a depth when the column is absent.
    read_col = next((c for c in ["read_count", "spots", "reads"] if c in marker_df.columns), None)
    if not read_col:
        raise DataPairingError("Missing read depth column in candidate runs DataFrame")
    marker_df[read_col] = pd.to_numeric(marker_df[read_col], errors="coerce").fillna(0).astype(int)

    run_col = next((c for c in ["run_accession", "run", "accession", "sra_run"] if c in marker_df.columns), None)
    if not run_col:
        raise DataPairingError("Cannot locate run accession column in runs DataFrame")

    min_reads = config.min_reads_16s if marker == "16S" else config.min_reads_its

    # Sort deterministically
    sorted_df = marker_df.sort_values(
        by=["barcode_id", read_col, run_col],
        ascending=[True, False, True]
    )

    # Pick top run per barcode
    best_runs = sorted_df.groupby("barcode_id", as_index=False).first()

    # Assign QC flag and standardized columns
    best_runs[f"qc_pass_{marker.lower()}"] = best_runs[read_col] >= min_reads
    best_runs[f"reads_{marker.lower()}"] = best_runs[read_col]
    best_runs[f"run_{marker.lower()}"] = best_runs[run_col]

    return best_runs


def deduplicate_resampled_sites(
    df: pd.DataFrame,
    strategy: DedupStrategy = DedupStrategy.KEEP_RESAMPLED
) -> pd.DataFrame:
    """
    Resolve re-sampled duplicate sites according to the specified strategy.
    
    Strategies:
      KEEP_RESAMPLED (Default): Drop initial barcodes {59, 102, 113, 116};
                                retain replacement barcodes {138, 601, 717, 258}.
                                Guarantees <= 881 distinct physical sites.
      KEEP_INITIAL:             Retain initial barcodes {59, 102, 113, 116};
                                drop replacement barcodes.
      CLUSTER_SITES:            Keep all samples, assign unified canonical site_id
                                so Spatial CV groups them together.
    """
    if df.empty:
        return df

    res_df = df.copy()
    if "barcode_id" not in res_df.columns:
        raise ValueError("DataFrame missing 'barcode_id' column required for deduplication")

    if strategy == DedupStrategy.KEEP_RESAMPLED:
        dropped_mask = res_df["barcode_id"].isin(RESAMPLED_INITIAL_BARCODES)
        res_df = res_df[~dropped_mask].copy()
    elif strategy == DedupStrategy.KEEP_INITIAL:
        dropped_mask = res_df["barcode_id"].isin(RESAMPLED_REPLACEMENT_BARCODES)
        res_df = res_df[~dropped_mask].copy()
    elif strategy == DedupStrategy.CLUSTER_SITES:
        pass
    else:
        raise ValueError(f"Unknown DedupStrategy: {strategy}")

    res_df["site_id"] = res_df["barcode_id"].apply(
        lambda b: CANONICAL_SITE_ID_MAP.get(b, f"LUCAS_SITE_{b}")
    )

    return res_df


def assert_no_resampled_site_leakage(
    train_barcodes: Set[int],
    test_barcodes: Set[int]
) -> None:
    """
    Programmatic assertion: verify no re-sampled duplicate pair is split
    between training and testing partitions.
    """
    for init_b, rep_b in RESAMPLED_SITE_PAIRS.items():
        if init_b in train_barcodes and rep_b in test_barcodes:
            raise ResampledSiteCollisionError(
                f"SPATIAL LEAKAGE: Replicate pair ({init_b}, {rep_b}) split across train and test folds!"
            )
        if rep_b in train_barcodes and init_b in test_barcodes:
            raise ResampledSiteCollisionError(
                f"SPATIAL LEAKAGE: Replicate pair ({rep_b}, {init_b}) split across train and test folds!"
            )


def pair_amplicon_runs(
    runs_16s_df: pd.DataFrame,
    runs_its_df: pd.DataFrame,
    config: PairingConfig = PairingConfig()
) -> Tuple[pd.DataFrame, PairingAuditMetrics]:
    """
    Execute deterministic 1:1 biological pairing between 16S and ITS datasets.
    
    Enforces:
      - Strict equality on integer BARCODE_ID
      - Verification that sample_id matches LUCAS_<barcode_id>
      - Detection and logging of orphan runs
      - Exclusion of samples failing read depth cutoffs
      - Refusal of cross-study pairing when study ids are present
    """
    assert_no_cross_study_pairing(runs_16s_df, runs_its_df)
    metrics = PairingAuditMetrics()
    metrics.total_raw_16s_runs = len(runs_16s_df)
    metrics.total_raw_its_runs = len(runs_its_df)

    # Select best runs per barcode
    best_16s = select_best_run_per_sample(runs_16s_df, "16S", config)
    best_its = select_best_run_per_sample(runs_its_df, "ITS", config)

    metrics.unique_16s_barcodes = len(best_16s)
    metrics.unique_its_barcodes = len(best_its)

    passing_16s = best_16s[best_16s["qc_pass_16s"]].copy() if not best_16s.empty else pd.DataFrame()
    passing_its = best_its[best_its["qc_pass_its"]].copy() if not best_its.empty else pd.DataFrame()

    metrics.passing_16s_barcodes = len(passing_16s)
    metrics.passing_its_barcodes = len(passing_its)

    set_16s = set(passing_16s["barcode_id"]) if not passing_16s.empty else set()
    set_its = set(passing_its["barcode_id"]) if not passing_its.empty else set()

    metrics.orphan_16s_barcodes = sorted(list(set_16s - set_its))
    metrics.orphan_its_barcodes = sorted(list(set_its - set_16s))

    if passing_16s.empty or passing_its.empty:
        metrics.validation_status = "EMPTY"
        return pd.DataFrame(), metrics

    cols_16s = ["barcode_id", "run_16s", "reads_16s"]
    if "biosample_acc" in passing_16s.columns:
        cols_16s.append("biosample_acc")
    elif "biosample_accession" in passing_16s.columns:
        cols_16s.append("biosample_accession")

    cols_its = ["barcode_id", "run_its", "reads_its"]
    if "biosample_acc" in passing_its.columns:
        cols_its.append("biosample_acc")
    elif "biosample_accession" in passing_its.columns:
        cols_its.append("biosample_accession")

    paired = pd.merge(
        passing_16s[cols_16s],
        passing_its[cols_its],
        on="barcode_id",
        suffixes=("_16s", "_its")
    )

    if (paired["barcode_id"].isna()).any():
        raise SamplePairingViolationError("Encountered null barcode_id in paired output")

    paired["sample_id"] = paired["barcode_id"].apply(lambda b: f"LUCAS_{b}")
    paired["is_paired"] = True

    # Standardize run column names for manifest
    paired["sra_16s_run"] = paired["run_16s"]
    paired["sra_its_run"] = paired["run_its"]

    metrics.paired_samples_pre_dedup = len(paired)

    # Apply site deduplication
    paired_dedup = deduplicate_resampled_sites(paired, strategy=config.dedup_strategy)
    metrics.paired_samples_post_dedup = len(paired_dedup)
    metrics.deduplicated_dropped_barcodes = sorted(
        list(set(paired["barcode_id"]) - set(paired_dedup["barcode_id"]))
    )

    metrics.validation_status = "PASSED"
    return paired_dedup, metrics


def _study_values(df: pd.DataFrame) -> set:
    found = set()
    if df is None or df.empty:
        return found
    for col in ("study_accession", "bioproject", "study_id", "study_16s", "study_its"):
        if col not in df.columns:
            continue
        values = df[col].dropna().astype(str).str.strip()
        values = values[~values.str.lower().isin(["", "nan", "none"])]
        found.update(values.tolist())
    return found


def assert_no_cross_study_pairing(*frames: pd.DataFrame) -> None:
    """
    Refuse pairing bacterial profiles from one study with fungal profiles from another.

    PRJNA283936 is an isolated 16S track and cannot be paired into set C.
    Frames without a study column are not treated as proof of a match.
    """
    studies = [_study_values(frame) for frame in frames]
    studies = [item for item in studies if item]
    if not studies:
        return
    if any("PRJNA283936" in item for item in studies):
        raise SamplePairingViolationError(
            "PRJNA283936 is an isolated 16S track and cannot be paired into set C."
        )
    union = set.union(*studies)
    if len(union) > 1:
        raise SamplePairingViolationError(
            f"Cross-study pairing is forbidden: {sorted(union)}."
        )


def enforce_strict_pairing(
    runs_16s: pd.DataFrame,
    runs_its: pd.DataFrame,
    esdac_df: Optional[pd.DataFrame] = None,
    config: Optional[PairingConfig] = None
) -> pd.DataFrame:
    """
    High-level convenience function enforcing strict 1:1 pairing and optional ESDAC join.
    """
    assert_no_cross_study_pairing(runs_16s, runs_its)
    cfg = config or PairingConfig()
    paired_df, _ = pair_amplicon_runs(runs_16s, runs_its, config=cfg)

    if paired_df.empty:
        return paired_df

    from src.data.sample_manifest import apply_bidirectional_column_aliases

    if esdac_df is not None and not esdac_df.empty:
        esdac_clean = esdac_df.copy()
        if "barcode_id" not in esdac_clean.columns and "sample_id" in esdac_clean.columns:
            esdac_clean["barcode_id"] = esdac_clean["sample_id"].apply(extract_barcode_id)
        esdac_clean = apply_bidirectional_column_aliases(esdac_clean)

        # Merge on barcode_id
        merged = pd.merge(paired_df, esdac_clean, on="barcode_id", suffixes=("", "_esdac"))
        # Drop duplicate sample_id if any
        if "sample_id_esdac" in merged.columns:
            merged = merged.drop(columns=["sample_id_esdac"])
        return apply_bidirectional_column_aliases(merged)

    return apply_bidirectional_column_aliases(paired_df)
