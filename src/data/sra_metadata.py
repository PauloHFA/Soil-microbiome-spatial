"""
SRA / ENA Metadata Ingestion and Schema Normalization Module.

Handles BioProject PRJNA952168 (LUCAS European soil biodiversity) and
complementary BioProject PRJNA283936 (Brazilian mangrove microbiome).
Provides schema normalization, molecular marker disambiguation (16S vs ITS),
BARCODE_ID extraction, quality threshold filtering, and local caching.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical schema columns for standardized SRA metadata DataFrame
CANONICAL_SRA_COLUMNS: List[str] = [
    "run_accession",
    "biosample_acc",
    "sample_title",
    "barcode_id",
    "marker",
    "platform",
    "instrument_model",
    "library_layout",
    "library_name",
    "read_count",
    "base_count",
    "fastq_urls",
    "is_primary",
    "qc_status",
]

# Audit threshold defaults from Labouyrie et al. 2023 / ESDAC
DEFAULT_MIN_READS_16S: int = 10_000
DEFAULT_MIN_READS_ITS: int = 500


def extract_barcode_id(
    sample_title: Optional[str] = None,
    library_name: Optional[str] = None,
    sample_name: Optional[str] = None,
) -> Optional[int]:
    """Extract integer BARCODE_ID (1-885) from SRA sample title or library name.

    Parameters
    ----------
    sample_title : Optional[str]
        Title string from SRA/ENA (e.g. 'LUCAS_92', 'LUCAS-365').
    library_name : Optional[str]
        Library name string (e.g. '92_HJVG7DRXX', 'ITS__92').
    sample_name : Optional[str]
        Alternative sample name field.

    Returns
    -------
    Optional[int]
        Extracted integer barcode, or None if unparseable.
    """
    candidates = [sample_title, library_name, sample_name]
    for text in candidates:
        if text is None or pd.isna(text):
            continue
        text_str = str(text).strip()
        if not text_str:
            continue

        # Pattern 1: Direct numeric
        if text_str.isdigit():
            return int(text_str)

        # Pattern 2: LUCAS_92 or LUCAS-92 or LUCAS 92
        match = re.search(r"LUCAS[_\-\s]+(\d+)", text_str, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Pattern 3: ITS__92 or ITS_92
        match = re.search(r"ITS[_\-]+(\d+)", text_str, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Pattern 4: 92_HJVG7DRXX (16S library format: <barcode>_<flowcell>)
        match = re.match(r"^(\d+)_", text_str)
        if match:
            return int(match.group(1))

        # Pattern 5: Any standalone 1 to 4 digit integer
        match = re.search(r"\b(\d{1,4})\b", text_str)
        if match:
            return int(match.group(1))

    return None


def infer_marker_type(
    platform: str = "",
    instrument_model: str = "",
    library_layout: str = "",
    library_name: str = "",
    library_strategy: str = "",
) -> Literal["16S", "ITS", "UNKNOWN"]:
    """Disambiguate molecular marker from sequencing platform, layout, and library metadata.

    Parameters
    ----------
    platform : str
        Sequencing platform (e.g. 'ILLUMINA', 'PACBIO_SMRT').
    instrument_model : str
        Instrument model (e.g. 'Illumina MiSeq', 'Sequel II').
    library_layout : str
        'PAIRED' or 'SINGLE'.
    library_name : str
        Library identifier string.
    library_strategy : str
        e.g. 'AMPLICON'.

    Returns
    -------
    Literal['16S', 'ITS', 'UNKNOWN']
        Classified molecular marker.
    """
    p_up = str(platform).strip().upper() if platform and not pd.isna(platform) else ""
    inst_up = str(instrument_model).strip().upper() if instrument_model and not pd.isna(instrument_model) else ""
    lib_up = str(library_name).strip().upper() if library_name and not pd.isna(library_name) else ""
    layout_up = str(library_layout).strip().upper() if library_layout and not pd.isna(library_layout) else ""

    # Rule 1: PacBio Sequel II is exclusively ITS in LUCAS 2018
    if "PACBIO" in p_up or "SEQUEL" in inst_up or lib_up.startswith("ITS"):
        return "ITS"

    # Rule 2: Illumina MiSeq paired-end is 16S V4-V5 in LUCAS 2018
    if "ILLUMINA" in p_up or "MISEQ" in inst_up or layout_up == "PAIRED":
        return "16S"

    # Fallback inspection of library name
    if re.match(r"^\d+_[A-Z0-9]+", lib_up):
        return "16S"

    return "UNKNOWN"


def normalize_sra_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw SRA/ENA dataframe headers and values to canonical internal schema.

    Supports both ENA filereport TSV and NCBI SRA RunInfo CSV schemas.
    """
    if df.empty:
        return pd.DataFrame(columns=CANONICAL_SRA_COLUMNS)

    norm_df = df.copy()
    rename_map: Dict[str, str] = {}
    for col in norm_df.columns:
        c_clean = str(col).strip().lower().replace(" ", "_")
        rename_map[col] = c_clean
    norm_df.rename(columns=rename_map, inplace=True)

    # Locate run accession column
    run_col = next((c for c in ["run_accession", "run", "accession", "sra_run"] if c in norm_df.columns), None)
    if not run_col:
        raise ValueError("Cannot locate run accession column in SRA metadata")

    biosample_col = next(
        (c for c in ["sample_accession", "biosample", "biosample_acc", "sample"] if c in norm_df.columns),
        None,
    )

    title_col = next(
        (c for c in ["sample_title", "samplename", "sample_name", "title"] if c in norm_df.columns),
        None,
    )

    lib_col = next((c for c in ["library_name", "libraryname"] if c in norm_df.columns), None)
    plat_col = next((c for c in ["instrument_platform", "platform"] if c in norm_df.columns), None)
    model_col = next((c for c in ["instrument_model", "model"] if c in norm_df.columns), None)
    layout_col = next((c for c in ["library_layout", "librarylayout", "layout"] if c in norm_df.columns), None)
    strategy_col = next((c for c in ["library_strategy", "librarystrategy"] if c in norm_df.columns), None)

    reads_col = next(
        (c for c in ["read_count", "spots", "readcount", "spots_with_mates"] if c in norm_df.columns),
        None,
    )

    bases_col = next((c for c in ["base_count", "bases", "basecount"] if c in norm_df.columns), None)

    fastq_col = next(
        (c for c in ["fastq_ftp", "download_path", "fastq_urls", "submitted_ftp"] if c in norm_df.columns),
        None,
    )

    result_rows: List[Dict[str, Any]] = []
    for _, row in norm_df.iterrows():
        run_acc = str(row[run_col]).strip()
        bio_acc = str(row[biosample_col]).strip() if biosample_col and pd.notna(row[biosample_col]) else ""
        title = str(row[title_col]).strip() if title_col and pd.notna(row[title_col]) else ""
        lib_name = str(row[lib_col]).strip() if lib_col and pd.notna(row[lib_col]) else ""
        platform = str(row[plat_col]).strip() if plat_col and pd.notna(row[plat_col]) else ""
        model = str(row[model_col]).strip() if model_col and pd.notna(row[model_col]) else ""
        layout = str(row[layout_col]).strip().upper() if layout_col and pd.notna(row[layout_col]) else ""
        strategy = str(row[strategy_col]).strip() if strategy_col and pd.notna(row[strategy_col]) else ""

        # Extract barcode
        b_id = extract_barcode_id(sample_title=title, library_name=lib_name)

        # Check explicit marker in columns first if available
        if "marker" in norm_df.columns and pd.notna(row["marker"]):
            m_val = str(row["marker"]).strip().upper()
            marker = "16S" if "16S" in m_val else ("ITS" if "ITS" in m_val else "UNKNOWN")
        else:
            marker = infer_marker_type(
                platform=platform,
                instrument_model=model,
                library_layout=layout,
                library_name=lib_name,
                library_strategy=strategy,
            )

        # Parse reads
        reads_val = 0
        if reads_col and pd.notna(row[reads_col]):
            try:
                reads_val = int(float(row[reads_col]))
            except (ValueError, TypeError):
                reads_val = 0

        bases_val = 0
        if bases_col and pd.notna(row[bases_col]):
            try:
                bases_val = int(float(row[bases_col]))
            except (ValueError, TypeError):
                bases_val = 0

        fastq_val = str(row[fastq_col]).strip() if fastq_col and pd.notna(row[fastq_col]) else ""

        result_rows.append(
            {
                "run_accession": run_acc,
                "biosample_acc": bio_acc,
                "sample_title": title,
                "barcode_id": b_id,
                "marker": marker,
                "platform": platform,
                "instrument_model": model,
                "library_layout": layout,
                "library_name": lib_name,
                "read_count": reads_val,
                "base_count": bases_val,
                "fastq_urls": fastq_val,
                "is_primary": True,
                "qc_status": "PASS",
            }
        )

    out_df = pd.DataFrame(result_rows)
    return out_df


class SRAMetadataParser:
    """Parser and quality auditor for SRA run metadata."""

    def __init__(
        self,
        min_reads_16s: int = DEFAULT_MIN_READS_16S,
        min_reads_its: int = DEFAULT_MIN_READS_ITS,
    ) -> None:
        self.min_reads_16s = min_reads_16s
        self.min_reads_its = min_reads_its

    def parse_and_audit(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize raw SRA metadata and audit sequencing depth thresholds."""
        norm_df = normalize_sra_dataframe(raw_df)
        if norm_df.empty:
            return norm_df

        # Audit QC status based on sequencing depth
        for idx, row in norm_df.iterrows():
            marker = row["marker"]
            reads = row["read_count"]
            barcode = row["barcode_id"]

            if barcode is None or pd.isna(barcode):
                norm_df.at[idx, "qc_status"] = "FAIL_MISSING_BARCODE"
            elif marker == "16S" and reads < self.min_reads_16s:
                norm_df.at[idx, "qc_status"] = "FAIL_LOW_DEPTH_16S"
            elif marker == "ITS" and reads < self.min_reads_its:
                norm_df.at[idx, "qc_status"] = "FAIL_LOW_DEPTH_ITS"
            elif marker == "UNKNOWN":
                norm_df.at[idx, "qc_status"] = "FAIL_UNKNOWN_MARKER"
            else:
                norm_df.at[idx, "qc_status"] = "PASS"

        return norm_df

    def select_primary_runs(
        self,
        audited_df: pd.DataFrame,
        policy: Literal["max_depth", "all"] = "max_depth",
    ) -> pd.DataFrame:
        """Resolve multiple sequencing runs per (barcode_id, marker).

        When multiple runs exist for the same physical barcode and marker
        (e.g. from flowcell re-sequencing passes), selects the run with the
        highest read depth as primary.
        """
        if audited_df.empty:
            return audited_df

        df = audited_df.copy()
        df["is_primary"] = False

        # Group by barcode_id and marker where barcode is valid and qc is PASS
        valid_mask = df["barcode_id"].notna() & (df["marker"].isin(["16S", "ITS"]))
        valid_df = df[valid_mask]

        primary_indices = []
        for (b_id, marker), group in valid_df.groupby(["barcode_id", "marker"]):
            best_idx = group["read_count"].idxmax()
            primary_indices.append(best_idx)

        df.loc[primary_indices, "is_primary"] = True
        return df


class SRAMetadataFetcher:
    """Fetcher for SRA / ENA run reports with local caching and offline fallback."""

    def __init__(
        self,
        cache_dir: Union[str, Path] = "data/raw",
        base_ena_url: str = "https://www.ebi.ac.uk/ena/portal/api/filereport",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_ena_url = base_ena_url

    def get_cache_path(self, bioproject: str) -> Path:
        """Return standardized path for cached SRA metadata file."""
        return self.cache_dir / f"sra_metadata_{bioproject}.csv"

    def fetch_or_load(
        self,
        bioproject: str = "PRJNA952168",
        force_download: bool = False,
    ) -> pd.DataFrame:
        """Fetch SRA metadata from ENA API or load from local cache.

        If network request fails or cannot be made, attempts to load cached file.
        """
        cache_file = self.get_cache_path(bioproject)

        if not force_download and cache_file.exists():
            logger.info("Loading SRA metadata for %s from cache: %s", bioproject, cache_file)
            return pd.read_csv(cache_file)

        logger.info("Attempting to retrieve SRA metadata for %s from ENA API", bioproject)
        try:
            import requests

            params = {
                "accession": bioproject,
                "result": "read_run",
                "format": "tsv",
                "download": "true",
                "fields": (
                    "study_accession,sample_accession,experiment_accession,"
                    "run_accession,sample_title,library_name,library_strategy,"
                    "library_source,library_selection,library_layout,"
                    "instrument_platform,instrument_model,read_count,"
                    "base_count,fastq_ftp"
                ),
            }
            resp = requests.get(self.base_ena_url, params=params, timeout=30)
            resp.raise_for_status()

            from io import StringIO

            raw_df = pd.read_csv(StringIO(resp.text), sep="\t")
            raw_df.to_csv(cache_file, index=False)
            logger.info("Successfully fetched and cached %d runs to %s", len(raw_df), cache_file)
            return raw_df

        except Exception as exc:
            if cache_file.exists():
                logger.warning("Network fetch failed (%s). Falling back to existing cache.", exc)
                return pd.read_csv(cache_file)
            logger.error("Failed to retrieve SRA metadata and no cache exists: %s", exc)
            raise RuntimeError(
                f"Cannot acquire SRA metadata for {bioproject} offline without pre-existing cache at {cache_file}."
            ) from exc
