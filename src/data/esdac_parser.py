"""
ESDAC LUCAS 2018 Soil Metadata Parser and Quality Auditor.

Parses European Soil Data Centre (ESDAC) edaphic, spatial, and bioclimatic
records. Normalizes column schemas, derives 3-class macro and 6-class fine
land cover targets (Labouyrie et al. 2023), audits physical parameter bounds,
and validates coordinate bounds under zero-leakage conventions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Valid bounding box for continental European LUCAS sites (WGS84)
EUROPE_LAT_MIN = 34.0
EUROPE_LAT_MAX = 72.0
EUROPE_LON_MIN = -12.0
EUROPE_LON_MAX = 35.0

# LUCAS 2018 Land Cover code mappings to the 6 core classes (Labouyrie et al. 2023)
NON_PERMANENT_CROP_CODES: Set[str] = {
    "B11", "B12", "B13", "B14", "B15", "B16", "B17", "B18", "B19",
    "B21", "B22", "B23",
    "B31", "B32", "B33", "B35", "B37",
    "B41", "B42", "B43", "B44", "B45",
    "B52", "B53", "B54", "B55",
}
PERMANENT_CROP_CODES: Set[str] = {
    "B71", "B72", "B73", "B74", "B75", "B77", "B81", "B82",
}
INTENSIVE_GRASS_CODES: Set[str] = {"E30"}
EXTENSIVE_GRASS_CODES: Set[str] = {"E10", "E20"}
BROADLEAVED_FOREST_CODES: Set[str] = {"C10", "C33"}
CONIFEROUS_FOREST_CODES: Set[str] = {"C21", "C22", "C23", "C31", "C32"}

# Physical boundary dictionary for edaphic and climatic variables
PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "latitude": (EUROPE_LAT_MIN, EUROPE_LAT_MAX),
    "longitude": (EUROPE_LON_MIN, EUROPE_LON_MAX),
    "ph_h2o": (3.0, 10.0),
    "ph_cacl2": (2.5, 9.5),
    "organic_carbon": (0.0, 600.0),       # g/kg
    "nitrogen": (0.0, 40.0),               # g/kg
    "c_n_ratio": (1.0, 80.0),              # dimensionless
    "phosphorus": (0.0, 500.0),            # mg/kg
    "potassium": (0.0, 3000.0),            # mg/kg
    "caco3": (0.0, 1000.0),                # g/kg
    "cec": (0.0, 150.0),                   # cmol(+)/kg
    "clay": (0.0, 100.0),                  # %
    "silt": (0.0, 100.0),                  # %
    "sand": (0.0, 100.0),                  # %
    "bulk_density": (0.2, 2.5),            # g/cm3
    "coarse_fragments": (0.0, 100.0),      # %
    "ec": (0.0, 1000.0),                   # mS/m
    "bio_1": (-15.0, 30.0),                # deg C
    "monthly_aridity": (0.0, 10.0),
    "isothermality": (0.0, 100.0),
    "temp_seasonality": (100.0, 2000.0),
    "annual_temp_range": (5.0, 60.0),
    "soil_moisture": (0.0, 1.0),           # m3/m3
}


def classify_land_cover(
    lc_code: Optional[str],
) -> Tuple[str, str, int, bool]:
    """Map raw LUCAS land cover code to macro and 6-class fine categories.

    Parameters
    ----------
    lc_code : Optional[str]
        Raw LUCAS land cover code (e.g. 'C21', 'B11', 'E10').

    Returns
    -------
    Tuple[str, str, int, bool]
        (macro_class, fine_class, fine_class_code, is_core_6class)
    """
    if not lc_code or not isinstance(lc_code, str):
        return ("Unknown", "Unknown", -1, False)

    code = lc_code.strip().upper()

    # Macro classification
    macro = "Other"
    if code.startswith("B"):
        macro = "Cropland"
    elif code.startswith("C"):
        macro = "Woodland"
    elif code.startswith("E"):
        macro = "Grassland"

    # Fine classification (6 core classes from Labouyrie et al. 2023)
    fine = "Other"
    fine_code = -1
    is_core = True

    if code in CONIFEROUS_FOREST_CODES:
        fine = "Coniferous woodland"
        fine_code = 0
    elif code in BROADLEAVED_FOREST_CODES:
        fine = "Broadleaved woodland"
        fine_code = 1
    elif code in EXTENSIVE_GRASS_CODES:
        fine = "Extensive grassland"
        fine_code = 2
    elif code in INTENSIVE_GRASS_CODES:
        fine = "Intensive grassland"
        fine_code = 3
    elif code in PERMANENT_CROP_CODES:
        fine = "Permanent cropland"
        fine_code = 4
    elif code in NON_PERMANENT_CROP_CODES:
        fine = "Non-permanent cropland"
        fine_code = 5
    else:
        fine = "Other"
        fine_code = -1
        is_core = False

    return (macro, fine, fine_code, is_core)


class ESDACLucasParser:
    """Parser and validator for ESDAC LUCAS 2018 Soil Biodiversity metadata."""

    COLUMN_ALIASES: Dict[str, List[str]] = {
        "barcode_id": ["barcode_id", "barcode", "id", "sample_id", "barcodeid"],
        "point_id": ["point_id", "pointid", "point", "lucas_point_id"],
        "latitude": ["gps_lat", "th_lat", "lat", "latitude"],
        "longitude": ["gps_long", "th_long", "long", "lon", "longitude"],
        "land_cover_raw": ["lc1", "lc", "land_cover", "landcover", "macro_land_cover"],
        "land_use_raw": ["lu1", "lu", "land_use", "landuse"],
        "ph_h2o": ["ph_in_h2o", "ph_h2o", "ph", "ph_water", "esdac_ph"],
        "ph_cacl2": ["ph_in_cacl2", "ph_cacl2", "ph_salt"],
        "organic_carbon": ["oc", "soc", "org_carb", "organic_carbon", "esdac_oc"],
        "nitrogen": ["n", "tot_n", "total_n", "total_nitrogen_content", "nitrogen", "esdac_n"],
        "c_n_ratio": ["c_n_ratio", "cn_ratio", "cn", "c_n"],
        "phosphorus": ["p", "olsen_p", "ext_p", "phosphorus", "phosphorus_content"],
        "potassium": ["k", "ext_k", "potassium", "extractable_potassium_content"],
        "caco3": ["caco3", "carbonates", "caco3_content", "carbonate_content"],
        "cec": ["cec", "cation_exchange"],
        "clay": ["clay", "clay_pct", "clay_fraction", "clay_content", "esdac_clay"],
        "silt": ["silt", "silt_pct", "silt_fraction", "silt_content", "esdac_silt"],
        "sand": ["sand", "sand_pct", "sand_fraction", "sand_content", "esdac_sand"],
        "bulk_density": ["bulk_density", "bd", "dry_bulk_density", "bulk_density_0_10_cm"],
        "coarse_fragments": ["coarse_fragments", "gravel", "coarse_pct"],
        "ec": ["ec", "electrical_conductivity"],
        "bio_1": ["bio_1", "mat", "mean_annual_temp"],
        "monthly_aridity": ["monthly_aridity", "aridity_index", "ai"],
        "isothermality": ["isothermality", "bio_3", "bio3"],
        "temp_seasonality": ["temperature_seasonality", "bio_4", "bio4"],
        "annual_temp_range": ["annual_temp_range", "bio_7", "bio7"],
        "soil_moisture": ["soil_moisture", "smap_moisture", "moisture"],
    }

    def __init__(self, filepath: Optional[Union[str, Path]] = None) -> None:
        self.filepath = Path(filepath) if filepath else None

    def read_table(self, path: Union[str, Path]) -> pd.DataFrame:
        """Read ESDAC file supporting CSV, TSV, or Excel (.xlsx/.xls)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"ESDAC metadata file not found: {p}")

        suffix = p.suffix.lower()
        if suffix in [".xlsx", ".xls"]:
            return pd.read_excel(p)
        elif suffix in [".tsv", ".tab"]:
            return pd.read_csv(p, sep="\t")
        elif suffix == ".csv":
            try:
                sample = p.read_text(encoding="utf-8", errors="ignore")[:2048]
                sep = ";" if sample.count(";") > sample.count(",") else ","
                return pd.read_csv(p, sep=sep)
            except Exception:
                return pd.read_csv(p)
        else:
            return pd.read_csv(p)

    def parse(self, filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Parse ESDAC table into a canonical, audited DataFrame."""
        target_path = Path(filepath) if filepath else self.filepath
        if not target_path:
            raise ValueError("No filepath provided for ESDAC parsing")

        raw_df = self.read_table(target_path)
        return self.parse_dataframe(raw_df)

    def parse_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw ESDAC DataFrame into canonical, typed schema."""
        if raw_df.empty:
            return pd.DataFrame()

        clean_cols = {col: str(col).strip().lower().replace(" ", "_") for col in raw_df.columns}
        df = raw_df.rename(columns=clean_cols).copy()

        # Build canonical mapped dictionary
        mapped_data: Dict[str, Any] = {}
        for canonical_name, aliases in self.COLUMN_ALIASES.items():
            matched_col = next((a for a in aliases if a in df.columns), None)
            if matched_col:
                mapped_data[canonical_name] = df[matched_col]
            else:
                mapped_data[canonical_name] = np.nan

        parsed_df = pd.DataFrame(mapped_data)

        # Barcode ID parsing
        parsed_df["barcode_id"] = pd.to_numeric(parsed_df["barcode_id"], errors="coerce")
        parsed_df = parsed_df.dropna(subset=["barcode_id"]).copy()
        parsed_df["barcode_id"] = parsed_df["barcode_id"].astype(int)

        # Point ID
        if parsed_df["point_id"].notna().any():
            parsed_df["point_id"] = pd.to_numeric(parsed_df["point_id"], errors="coerce").fillna(-1).astype(int)
        else:
            parsed_df["point_id"] = 100000 + parsed_df["barcode_id"]

        # Ensure numeric types for edaphic and spatial variables
        numeric_cols = [
            "latitude", "longitude", "ph_h2o", "ph_cacl2", "organic_carbon",
            "nitrogen", "c_n_ratio", "phosphorus", "potassium", "caco3",
            "cec", "clay", "silt", "sand", "bulk_density", "coarse_fragments",
            "ec", "bio_1", "monthly_aridity", "isothermality", "temp_seasonality",
            "annual_temp_range", "soil_moisture",
        ]
        for col in numeric_cols:
            if col in parsed_df.columns:
                parsed_df[col] = pd.to_numeric(parsed_df[col], errors="coerce")

        # Calculate C/N ratio if missing and components are present
        missing_cn = parsed_df["c_n_ratio"].isna() & parsed_df["organic_carbon"].notna() & parsed_df["nitrogen"].notna()
        valid_n = parsed_df["nitrogen"] > 0
        fill_mask = missing_cn & valid_n
        parsed_df.loc[fill_mask, "c_n_ratio"] = (
            parsed_df.loc[fill_mask, "organic_carbon"] / parsed_df.loc[fill_mask, "nitrogen"]
        )

        # Map land cover classes
        macros = []
        fines = []
        fine_codes = []
        is_cores = []

        for _, row in parsed_df.iterrows():
            lc = row["land_cover_raw"]
            if pd.isna(lc) and "macro_land_cover" in df.columns:
                macro_val = str(row.get("macro_land_cover", "Other")).strip()
                macros.append(macro_val)
                fines.append("Other")
                fine_codes.append(-1)
                is_cores.append(False)
            else:
                macro, fine, f_code, is_c = classify_land_cover(lc)
                macros.append(macro)
                fines.append(fine)
                fine_codes.append(f_code)
                is_cores.append(is_c)

        parsed_df["macro_land_cover"] = macros
        parsed_df["fine_land_cover"] = fines
        parsed_df["fine_land_cover_code"] = fine_codes
        parsed_df["is_core_6class"] = is_cores

        # Add sample_id string for direct relational joins (e.g. 'LUCAS_92')
        parsed_df["sample_id"] = "LUCAS_" + parsed_df["barcode_id"].astype(str)

        return parsed_df

    def audit_physical_bounds(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate parsed DataFrame against physical and spatial constraints."""
        report: Dict[str, Any] = {
            "total_records": len(df),
            "out_of_bounds_counts": {},
            "texture_violation_count": 0,
            "core_6class_count": int(df["is_core_6class"].sum()) if "is_core_6class" in df.columns else 0,
            "complete_records_count": 0,
        }

        for var, (min_val, max_val) in PHYSICAL_BOUNDS.items():
            if var in df.columns:
                series = df[var].dropna()
                oob = ((series < min_val) | (series > max_val)).sum()
                report["out_of_bounds_counts"][var] = int(oob)

        # Soil texture conservation: clay + silt + sand ~ 100%
        if all(k in df.columns for k in ["clay", "silt", "sand"]):
            tex_df = df[["clay", "silt", "sand"]].dropna()
            if not tex_df.empty:
                sums = tex_df.sum(axis=1)
                violations = ((sums < 95.0) | (sums > 105.0)).sum()
                report["texture_violation_count"] = int(violations)

        # Check records with complete essential edaphic features
        essential = ["latitude", "longitude", "ph_h2o", "organic_carbon", "nitrogen", "clay", "silt", "sand"]
        present_ess = [c for c in essential if c in df.columns]
        if len(present_ess) == len(essential):
            complete = df[present_ess].dropna()
            report["complete_records_count"] = len(complete)

        return report
