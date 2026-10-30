"""
scientific_cohort.py - Gate for the revised-spec paired LUCAS/ESDAC comparison.

Mock rows are not a scientific cohort. Cross-study pairing and PRJNA283936
are refused before any integrated model is fit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.pairing_engine import (
    SamplePairingViolationError,
    assert_no_cross_study_pairing,
)


LUCAS_STUDY_ACCESSION = "PRJNA952168"
MANGROVE_STUDY_ACCESSION = "PRJNA283936"

# Identity and spatial fields required to claim the revised comparison.
# A single missing edaphic measure does not fail the cohort; the ESDAC join
# is evidenced by at least one esdac_* column, not by a complete chemistry panel.
IDENTITY_COLUMNS: Tuple[str, ...] = (
    "sample_id",
    "site_id",
    "latitude",
    "longitude",
    "sra_16s_run",
    "sra_its_run",
    "study_accession",
)


class ScientificCohortError(Exception):
    """Raised when the files on disk cannot support the revised comparison."""


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _has_esdac_join(df: pd.DataFrame) -> bool:
    esdac_cols = [col for col in df.columns if col.startswith("esdac_")]
    if not esdac_cols:
        return False
    return bool(df[esdac_cols].notna().any().any())


def validate_paired_lucas_esdac_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept only a paired LUCAS/ESDAC manifest for sets A, B and C.

    Refuses missing identity fields, cross-study rows, mangrove rows and
    samples that do not carry both markers.
    """
    if df is None or df.empty:
        raise ScientificCohortError(
            "Paired LUCAS/ESDAC manifest is empty. Mock data is not a scientific cohort."
        )

    missing = [col for col in IDENTITY_COLUMNS if col not in df.columns]
    if missing:
        raise ScientificCohortError(
            "Scientific comparison requires an audited paired LUCAS/ESDAC manifest. "
            f"Missing columns: {missing}. Synthetic files are not stamped as PRJNA952168."
        )

    label_col = _label_column(df, preferred=None)
    if label_col is None:
        raise ScientificCohortError(
            "Manifest has no environmental category column "
            "(fine_land_cover, macro_land_cover or land_cover)."
        )

    if not _has_esdac_join(df):
        raise ScientificCohortError(
            "Paired comparison requires an ESDAC join (at least one esdac_* column). "
            "A missing single variable is allowed; a missing join is not."
        )

    cohort = df.copy()
    cohort["study_accession"] = _clean_text(cohort["study_accession"])
    if (cohort["study_accession"] == "").any() or cohort["study_accession"].isna().any():
        raise ScientificCohortError("study_accession cannot be blank.")

    try:
        assert_no_cross_study_pairing(cohort)
    except SamplePairingViolationError as exc:
        raise ScientificCohortError(str(exc)) from exc

    studies = set(cohort["study_accession"].unique())
    if studies != {LUCAS_STUDY_ACCESSION}:
        raise ScientificCohortError(
            "Integrated comparison accepts only "
            f"{LUCAS_STUDY_ACCESSION}. Found: {sorted(studies)}."
        )

    for run_col in ("sra_16s_run", "sra_its_run"):
        runs = _clean_text(cohort[run_col])
        if runs.isna().any() or (runs == "").any() or (runs.str.lower() == "nan").any():
            raise ScientificCohortError(
                f"{run_col} must be present on every paired sample. "
                "A one-marker sample cannot enter set C."
            )
        cohort[run_col] = runs

    if "study_16s" in cohort.columns or "study_its" in cohort.columns:
        try:
            assert_no_cross_study_pairing(cohort)
        except SamplePairingViolationError as exc:
            raise ScientificCohortError(str(exc)) from exc
        for col in ("study_16s", "study_its"):
            if col in cohort.columns and (cohort[col].astype(str) != LUCAS_STUDY_ACCESSION).any():
                raise ScientificCohortError(
                    f"{col} disagrees with the paired LUCAS study. Cross-study pairing is forbidden."
                )

    sample_ids = _clean_text(cohort["sample_id"])
    if sample_ids.duplicated().any():
        raise ScientificCohortError(
            "Duplicate sample_id. Technical duplicates are not independent observations."
        )
    if sample_ids.str.startswith("MANGROVE_").any():
        raise ScientificCohortError(
            "Mangrove sample ids cannot enter the integrated LUCAS comparison."
        )

    if cohort["site_id"].isna().any():
        raise ScientificCohortError("site_id is required for spatial blocking.")
    if cohort["latitude"].isna().any() or cohort["longitude"].isna().any():
        raise ScientificCohortError(
            "Coordinates are required. Without them the run must not claim spatial validation."
        )

    return cohort


def _label_column(df: pd.DataFrame, preferred: Optional[str]) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    for col in ("fine_land_cover", "macro_land_cover", "land_cover"):
        if col in df.columns:
            return col
    return None


def resolve_target_column(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    """Prefer the revised fine land-cover categories when that column exists."""
    column = _label_column(df, preferred)
    if column is None:
        raise ScientificCohortError("No environmental category column is available.")
    if df[column].isna().any() or (df[column].astype(str).str.strip() == "").any():
        raise ScientificCohortError(f"Category column '{column}' contains blank labels.")
    return column


def load_scientific_cohort(
    data_dir: str | Path,
    preferred_label: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
    """
    Load the paired manifest and both count matrices.

    Does not create files and does not call the mock generator.
    """
    root = Path(data_dir)
    manifest_path = root / "paired_sample_manifest.csv"
    path_16s = root / "bacteria_16s_asv_matrix.csv"
    path_its = root / "fungi_its_asv_matrix.csv"
    missing = [str(path) for path in (manifest_path, path_16s, path_its) if not path.exists()]
    if missing:
        raise ScientificCohortError(
            "Scientific flow requires a paired LUCAS/ESDAC manifest and both count matrices. "
            f"Missing: {missing}. Mock generation is test-only and was not run."
        )

    from src.bioinformatics.count_matrix import load_count_matrix

    manifest = validate_paired_lucas_esdac_manifest(pd.read_csv(manifest_path))
    label_col = resolve_target_column(manifest, preferred_label)
    counts_16s, ids_16s, _ = load_count_matrix(path_16s)
    counts_its, ids_its, _ = load_count_matrix(path_its)
    counts_16s = _align_counts(counts_16s, ids_16s, manifest["sample_id"], path_16s.name)
    counts_its = _align_counts(counts_its, ids_its, manifest["sample_id"], path_its.name)
    if counts_16s.shape[0] != counts_its.shape[0]:
        raise ScientificCohortError("16S and ITS matrices do not have the same paired samples.")
    return manifest, counts_16s, counts_its, label_col


def _align_counts(
    counts: np.ndarray,
    sample_ids: Sequence[str],
    manifest_ids: Iterable[str],
    matrix_name: str,
) -> np.ndarray:
    wanted = [str(sample_id) for sample_id in manifest_ids]
    index = {str(sample_id): pos for pos, sample_id in enumerate(sample_ids)}
    missing = [sample_id for sample_id in wanted if sample_id not in index]
    if missing:
        raise ScientificCohortError(
            f"{matrix_name} is missing {len(missing)} paired sample ids, including {missing[:3]}."
        )
    order = [index[sample_id] for sample_id in wanted]
    return np.asarray(counts)[order]
