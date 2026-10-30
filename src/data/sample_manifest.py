"""
src/data/sample_manifest.py
Audited paired sample manifest builder, validator, and exporter.

Enforces strict column schema, biological and spatial boundary checks,
relational integrity, and audit report generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

MANDATORY_MANIFEST_COLUMNS: List[str] = [
    "sample_id",
    "barcode_id",
    "site_id",
    "latitude",
    "longitude",
    "land_cover",
    "land_cover_code",
    "sra_16s_run",
    "sra_its_run",
    "esdac_ph",
    "esdac_oc",
    "esdac_n",
    "esdac_clay",
    "esdac_silt",
    "esdac_sand",
]

LAND_COVER_MAP: Dict[str, int] = {
    "Cropland": 0,
    "Grassland": 1,
    "Woodland": 2,
}

# ESDAC parser names <-> paired-manifest names. Fill only the missing side.
ESDAC_MANIFEST_ALIASES: Dict[str, str] = {
    "ph_h2o": "esdac_ph",
    "organic_carbon": "esdac_oc",
    "nitrogen": "esdac_n",
    "clay": "esdac_clay",
    "silt": "esdac_silt",
    "sand": "esdac_sand",
    "fine_land_cover_code": "land_cover_code",
    "macro_land_cover": "land_cover",
}


def apply_bidirectional_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Copy values across ESDAC and manifest column names when one side is absent."""
    if df is None or df.empty:
        return df.copy() if df is not None else df

    out = df.copy()
    for left, right in ESDAC_MANIFEST_ALIASES.items():
        if left in out.columns and right not in out.columns:
            out[right] = out[left]
        elif right in out.columns and left not in out.columns:
            out[left] = out[right]
    return out


class ManifestSchemaValidationError(Exception):
    """Raised when manifest dataframe fails schema or value bounds validation."""
    pass


def validate_manifest_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate paired sample manifest against schema constraints.
    
    Raises ManifestSchemaValidationError upon any violation.
    Returns audit summary dict on success.
    """
    df = apply_bidirectional_column_aliases(df)
    errors: List[str] = []

    missing_cols = [c for c in MANDATORY_MANIFEST_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing mandatory columns: {missing_cols}")

    if errors:
        raise ManifestSchemaValidationError("; ".join(errors))

    # 2. Non-null assertions on mandatory fields
    for col in MANDATORY_MANIFEST_COLUMNS:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            errors.append(f"Column '{col}' contains {null_count} null values")

    # 3. Barcode uniqueness and 1:1 pairing
    if df["barcode_id"].duplicated().any():
        dups = df[df["barcode_id"].duplicated()]["barcode_id"].tolist()
        errors.append(f"Duplicate barcode_ids detected in manifest: {dups}")

    if df["sample_id"].duplicated().any():
        errors.append("Duplicate sample_ids detected in manifest")

    # 4. Coordinate boundary assertions
    if not df["latitude"].between(-90.0, 90.0).all():
        errors.append("Latitude out of bounds [-90, 90]")
    if not df["longitude"].between(-180.0, 180.0).all():
        errors.append("Longitude out of bounds [-180, 180]")

    # 5. Edaphic property boundary assertions
    if not df["esdac_ph"].between(2.0, 11.0).all():
        errors.append("esdac_ph out of biological bounds [2.0, 11.0]")
    if (df["esdac_oc"] < 0.0).any():
        errors.append("Negative values found in esdac_oc")
    if (df["esdac_n"] < 0.0).any():
        errors.append("Negative values found in esdac_n")

    # 6. Soil texture sanity check (clay + silt + sand <= 105%)
    texture_sum = df["esdac_clay"] + df["esdac_silt"] + df["esdac_sand"]
    if (texture_sum > 105.0).any():
        errors.append("Soil texture sum (clay + silt + sand) exceeds 105%")

    # 7. SRA run format check
    invalid_16s = ~df["sra_16s_run"].astype(str).str.match(r"^SRR\d+$", na=False)
    if invalid_16s.any():
        bad = df.loc[invalid_16s, "sra_16s_run"].tolist()
        errors.append(f"Invalid sra_16s_run accessions: {bad}")

    invalid_its = ~df["sra_its_run"].astype(str).str.match(r"^SRR\d+$", na=False)
    if invalid_its.any():
        bad = df.loc[invalid_its, "sra_its_run"].tolist()
        errors.append(f"Invalid sra_its_run accessions: {bad}")

    # 8. Land cover categories
    valid_classes = set(LAND_COVER_MAP.keys())
    invalid_classes = set(df["land_cover"]) - valid_classes
    if invalid_classes:
        errors.append(f"Invalid land_cover categories found: {invalid_classes}")

    if errors:
        raise ManifestSchemaValidationError("\n".join(errors))

    return {
        "status": "VALID",
        "sample_count": len(df),
        "unique_sites": int(df["site_id"].nunique()),
        "class_distribution": df["land_cover"].value_counts().to_dict(),
        "mean_ph": float(df["esdac_ph"].mean()),
        "ph_range": [float(df["esdac_ph"].min()), float(df["esdac_ph"].max())],
    }


def save_paired_manifest(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    validate: bool = True
) -> None:
    """Validate and save paired sample manifest to CSV."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if validate:
        validate_manifest_schema(df)

    df.to_csv(out_path, index=False, encoding="utf-8")


def load_paired_manifest(filepath: Union[str, Path], validate: bool = True) -> pd.DataFrame:
    """Load paired sample manifest from CSV and optionally validate schema."""
    in_path = Path(filepath)
    if not in_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {in_path}")

    df = pd.read_csv(in_path)
    if validate:
        validate_manifest_schema(df)

    return df


def export_manifest_audit_report(
    manifest_df: pd.DataFrame,
    report_path: Union[str, Path]
) -> Dict[str, Any]:
    """Export detailed JSON audit metrics for the sample manifest."""
    summary = validate_manifest_schema(manifest_df)

    summary["edaphic_summary"] = {
        "ph": {
            "mean": float(manifest_df["esdac_ph"].mean()),
            "std": float(manifest_df["esdac_ph"].std()),
            "min": float(manifest_df["esdac_ph"].min()),
            "max": float(manifest_df["esdac_ph"].max()),
        },
        "oc": {
            "mean": float(manifest_df["esdac_oc"].mean()),
            "std": float(manifest_df["esdac_oc"].std()),
            "min": float(manifest_df["esdac_oc"].min()),
            "max": float(manifest_df["esdac_oc"].max()),
        },
        "n": {
            "mean": float(manifest_df["esdac_n"].mean()),
            "std": float(manifest_df["esdac_n"].std()),
            "min": float(manifest_df["esdac_n"].min()),
            "max": float(manifest_df["esdac_n"].max()),
        },
    }

    out_file = Path(report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
