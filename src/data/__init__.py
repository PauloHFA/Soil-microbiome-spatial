"""
src/data/__init__.py
Milestone 1 Data Ingestion, Auditing & Pairing Subsystem.

Exports public data ingestion classes, parsers, pairing engine, manifest builders,
and mock data generators.
"""

from src.data.sra_metadata import (
    CANONICAL_SRA_COLUMNS,
    DEFAULT_MIN_READS_16S,
    DEFAULT_MIN_READS_ITS,
    SRAMetadataFetcher,
    SRAMetadataParser,
    extract_barcode_id,
    infer_marker_type,
    normalize_sra_dataframe,
)
from src.data.esdac_parser import (
    ESDACLucasParser,
    classify_land_cover,
    PHYSICAL_BOUNDS,
)
from src.data.pairing_engine import (
    CANONICAL_SITE_ID_MAP,
    RESAMPLED_SITE_PAIRS,
    DedupStrategy,
    PairingConfig,
    PairingStrategy,
    PairingAuditMetrics,
    DataPairingError,
    SamplePairingViolationError,
    ResampledSiteCollisionError,
    InsufficientReadsError,
    deduplicate_resampled_sites,
    pair_amplicon_runs,
    enforce_strict_pairing,
    assert_no_resampled_site_leakage,
    assert_no_cross_study_pairing,
    select_best_run_per_sample,
)
from src.data.sample_manifest import (
    MANDATORY_MANIFEST_COLUMNS,
    LAND_COVER_MAP,
    ManifestSchemaValidationError,
    validate_manifest_schema,
    save_paired_manifest,
    load_paired_manifest,
    export_manifest_audit_report,
)
from src.data.mock_generator import (
    EMPIRICAL_CLASS_DISTRIBUTION,
    PRIMERS,
    MockGeneratorConfig,
    MockDataGenerator,
    generate_mock_data,
)
from src.data.mangrove_ingestion import (
    MANGROVE_ESTUARIES,
    MANGROVE_MANIFEST_COLUMNS,
    MangroveDatasetIngestion,
    assert_mangrove_isolation,
)

__all__ = [
    "CANONICAL_SRA_COLUMNS",
    "DEFAULT_MIN_READS_16S",
    "DEFAULT_MIN_READS_ITS",
    "SRAMetadataFetcher",
    "SRAMetadataParser",
    "extract_barcode_id",
    "infer_marker_type",
    "normalize_sra_dataframe",
    "ESDACLucasParser",
    "classify_land_cover",
    "PHYSICAL_BOUNDS",
    "CANONICAL_SITE_ID_MAP",
    "RESAMPLED_SITE_PAIRS",
    "DedupStrategy",
    "PairingConfig",
    "PairingStrategy",
    "PairingAuditMetrics",
    "DataPairingError",
    "SamplePairingViolationError",
    "ResampledSiteCollisionError",
    "InsufficientReadsError",
    "deduplicate_resampled_sites",
    "pair_amplicon_runs",
    "enforce_strict_pairing",
    "assert_no_resampled_site_leakage",
    "assert_no_cross_study_pairing",
    "select_best_run_per_sample",
    "MANDATORY_MANIFEST_COLUMNS",
    "LAND_COVER_MAP",
    "ManifestSchemaValidationError",
    "validate_manifest_schema",
    "save_paired_manifest",
    "load_paired_manifest",
    "export_manifest_audit_report",
    "EMPIRICAL_CLASS_DISTRIBUTION",
    "PRIMERS",
    "MockGeneratorConfig",
    "MockDataGenerator",
    "generate_mock_data",
    "MANGROVE_ESTUARIES",
    "MANGROVE_MANIFEST_COLUMNS",
    "MangroveDatasetIngestion",
    "assert_mangrove_isolation",
]
