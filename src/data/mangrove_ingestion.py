"""
Isolated Ingestion Sub-Pipeline for Brazilian Mangrove Microbiomes (PRJNA283936).

BioProject PRJNA283936 (Tavares et al. 2021) covers 105 semi-arid mangrove
samples (16S V3-V4 only) from 7 Brazilian estuaries.
This module maintains strict technical and statistical isolation from the
primary LUCAS multi-marker dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The 7 semi-arid mangrove estuaries evaluated in Tavares et al. 2021 / test suite
MANGROVE_ESTUARIES: List[str] = [
    "Ceara",
    "Cocó",
    "Pacoti",
    "Choró",
    "Pirangi",
    "Jaguaribe",
    "Acu",
]

# Alternate spellings and additional semi-arid estuaries
MANGROVE_ESTUARIES_ALIASES: Dict[str, str] = {
    "ceara": "Ceara",
    "ceará": "Ceara",
    "coco": "Cocó",
    "cocó": "Cocó",
    "pacoti": "Pacoti",
    "choro": "Choró",
    "choró": "Choró",
    "pirangi": "Pirangi",
    "jaguaribe": "Jaguaribe",
    "acu": "Acu",
    "açu": "Acu",
    "acarau": "Acarau",
    "acáraú": "Acarau",
    "coreau": "Coreau",
    "coreaú": "Coreau",
    "timonha": "Timonha",
}

# Canonical columns for isolated mangrove manifest
MANGROVE_MANIFEST_COLUMNS: List[str] = [
    "sample_id",
    "run_accession",
    "biosample_acc",
    "bioproject",
    "estuary",
    "sampling_station",
    "latitude",
    "longitude",
    "salinity_psu",
    "ph",
    "organic_matter_pct",
    "depth_cm",
    "read_count",
    "marker",
    "target_region",
    "biome",
]


class MangroveDatasetIngestion:
    """Ingestion, metadata extraction, and manifest builder for PRJNA283936."""

    def __init__(
        self,
        output_dir: Union[str, Path] = "data/processed",
        cache_dir: Union[str, Path] = "data/raw",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def parse_mangrove_sra_metadata(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Parse raw SRA run info for PRJNA283936 into standardized format."""
        if raw_df.empty:
            return pd.DataFrame(columns=MANGROVE_MANIFEST_COLUMNS)

        records: List[Dict[str, Any]] = []
        for _, row in raw_df.iterrows():
            # Extract run accession
            run_col = next((c for c in ["Run", "run_accession", "run", "accession"] if c in raw_df.columns), None)
            run_acc = str(row[run_col]).strip() if run_col and pd.notna(row[run_col]) else ""

            # Extract BioSample
            bio_col = next((c for c in ["BioSample", "sample_accession", "biosample"] if c in raw_df.columns), None)
            bio_acc = str(row[bio_col]).strip() if bio_col and pd.notna(row[bio_col]) else ""

            # Extract sample title
            title_col = next((c for c in ["Sample_Title", "sample_title", "SampleName", "sample_id"] if c in raw_df.columns), None)
            title = str(row[title_col]).strip() if title_col and pd.notna(row[title_col]) else f"MANG_{run_acc}"

            # Estuary parsing from title or estuary column
            estuary = "Unknown"
            if "estuary" in raw_df.columns and pd.notna(row["estuary"]):
                est_val = str(row["estuary"]).strip()
                est_clean = MANGROVE_ESTUARIES_ALIASES.get(est_val.lower(), est_val)
                estuary = est_clean
            else:
                title_lower = title.lower()
                for alias_k, canonical_v in MANGROVE_ESTUARIES_ALIASES.items():
                    if alias_k in title_lower:
                        estuary = canonical_v
                        break

            # Read count
            reads_col = next((c for c in ["spots", "read_count", "reads"] if c in raw_df.columns), None)
            read_cnt = int(float(row[reads_col])) if reads_col and pd.notna(row[reads_col]) else 0

            # Latitude & Longitude
            lat_col = next((c for c in ["latitude", "lat"] if c in raw_df.columns), None)
            lat = float(row[lat_col]) if lat_col and pd.notna(row[lat_col]) else -3.5

            lon_col = next((c for c in ["longitude", "long", "lon"] if c in raw_df.columns), None)
            lon = float(row[lon_col]) if lon_col and pd.notna(row[lon_col]) else -39.0

            sal_col = next((c for c in ["salinity_psu", "salinity"] if c in raw_df.columns), None)
            sal = float(row[sal_col]) if sal_col and pd.notna(row[sal_col]) else np.nan

            ph_col = next((c for c in ["ph", "ph_h2o"] if c in raw_df.columns), None)
            ph = float(row[ph_col]) if ph_col and pd.notna(row[ph_col]) else np.nan

            om_col = next((c for c in ["organic_matter_pct", "organic_matter"] if c in raw_df.columns), None)
            om = float(row[om_col]) if om_col and pd.notna(row[om_col]) else np.nan

            records.append(
                {
                    "sample_id": title,
                    "run_accession": run_acc,
                    "biosample_acc": bio_acc,
                    "bioproject": "PRJNA283936",
                    "estuary": estuary,
                    "sampling_station": str(row.get("sampling_station", "Station_1")),
                    "latitude": lat,
                    "longitude": lon,
                    "salinity_psu": sal,
                    "ph": ph,
                    "organic_matter_pct": om,
                    "depth_cm": float(row.get("depth_cm", 10.0)),
                    "read_count": read_cnt,
                    "marker": "16S",
                    "target_region": "V3-V4",
                    "biome": "Tropical Mangrove",
                }
            )

        return pd.DataFrame(records)

    def generate_synthetic_mangrove_manifest(
        self,
        n_samples: int = 105,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate deterministic synthetic mock mangrove manifest for testing."""
        rng = np.random.RandomState(seed)
        rows: List[Dict[str, Any]] = []

        samples_per_estuary = n_samples // len(MANGROVE_ESTUARIES)
        idx = 1
        for estuary in MANGROVE_ESTUARIES:
            base_lat = -3.0 - rng.uniform(0.1, 2.0)
            base_lon = -38.0 - rng.uniform(0.1, 2.5)

            for station_num in range(1, samples_per_estuary + 1):
                clean_name = estuary.replace("ó", "o")
                s_id = f"MANG_{clean_name.upper()}_S{station_num:02d}"
                run_acc = f"SRR202{idx:04d}"
                bio_acc = f"SAMN037{idx:05d}"

                salinity = float(rng.uniform(5.0, 42.0))
                ph = float(rng.uniform(6.5, 8.4))
                om = float(rng.uniform(1.2, 18.5))
                reads = int(rng.randint(15_000, 85_000))

                rows.append(
                    {
                        "sample_id": s_id,
                        "run_accession": run_acc,
                        "biosample_acc": bio_acc,
                        "bioproject": "PRJNA283936",
                        "estuary": estuary,
                        "sampling_station": f"S{station_num:02d}",
                        "latitude": round(base_lat + rng.uniform(-0.05, 0.05), 4),
                        "longitude": round(base_lon + rng.uniform(-0.05, 0.05), 4),
                        "salinity_psu": round(salinity, 2),
                        "ph": round(ph, 2),
                        "organic_matter_pct": round(om, 2),
                        "depth_cm": float(rng.choice([5.0, 10.0, 20.0])),
                        "read_count": reads,
                        "marker": "16S",
                        "target_region": "V3-V4",
                        "biome": "Tropical Mangrove",
                    }
                )
                idx += 1

        df = pd.DataFrame(rows)
        return df

    def export_manifest(self, df: pd.DataFrame, filename: str = "mangrove_prjna283936_manifest.csv") -> Path:
        """Export isolated mangrove manifest to data/processed directory."""
        out_path = self.output_dir / filename
        df.to_csv(out_path, index=False)
        logger.info("Exported isolated mangrove manifest to %s (%d records)", out_path, len(df))
        return out_path


def assert_mangrove_isolation(
    lucas_manifest: pd.DataFrame,
    mangrove_manifest: pd.DataFrame,
) -> None:
    """Programmatically verify complete mathematical isolation of mangrove dataset.

    Enforces:
    1. Zero overlap in sample identifiers.
    2. Zero overlap in SRA run accessions.
    3. Mangrove contains solely 16S markers (zero ITS).
    4. Geographic disjunction (Europe vs Brazilian equatorial coast).
    """
    # 1. Sample ID disjointness
    lucas_samples = set(lucas_manifest["sample_id"].astype(str))
    mangrove_samples = set(mangrove_manifest["sample_id"].astype(str))
    sample_overlap = lucas_samples.intersection(mangrove_samples)
    assert len(sample_overlap) == 0, (
        f"CRITICAL ISOLATION BREACH: Sample IDs shared between LUCAS and Mangrove: {sample_overlap}"
    )

    # 2. Run Accession disjointness
    lucas_runs_16s = set(lucas_manifest["sra_16s_run"].dropna().astype(str)) if "sra_16s_run" in lucas_manifest else set()
    lucas_runs_its = set(lucas_manifest["sra_its_run"].dropna().astype(str)) if "sra_its_run" in lucas_manifest else set()
    all_lucas_runs = lucas_runs_16s.union(lucas_runs_its)

    mangrove_runs = set(mangrove_manifest["run_accession"].dropna().astype(str))
    run_overlap = all_lucas_runs.intersection(mangrove_runs)
    assert len(run_overlap) == 0, (
        f"CRITICAL ISOLATION BREACH: Run accessions shared: {run_overlap}"
    )

    # 3. Marker homogeneity check: Mangrove must be 16S only
    markers = set(mangrove_manifest["marker"].dropna().unique())
    assert markers == {"16S"}, f"Mangrove dataset must only contain 16S marker, found: {markers}"

    # 4. Latitude disjunction: LUCAS > 30 deg N; Mangrove < 0 deg (Southern hemisphere)
    if "latitude" in lucas_manifest and "latitude" in mangrove_manifest:
        min_lucas_lat = lucas_manifest["latitude"].min()
        max_mang_lat = mangrove_manifest["latitude"].max()
        assert min_lucas_lat > 30.0, f"LUCAS latitude out of European range: {min_lucas_lat}"
        assert max_mang_lat < 0.0, f"Mangrove latitude out of Southern Hemisphere range: {max_mang_lat}"
