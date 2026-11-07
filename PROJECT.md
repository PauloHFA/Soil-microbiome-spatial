# Project: Soil Microbiome Environmental Discrimination Pipeline under Spatial Cross-Validation

## Architecture

The project implements an end-to-end computational pipeline for postdoctoral research evaluating the discriminatory potential of bacterial (16S rRNA), fungal (ITS), and integrated soil profiles from public databases (BioProject PRJNA952168 / LUCAS Soil Survey, ESDAC edaphic metadata, and PRJNA283936) under spatial cross-validation.

### Modular Architecture
```
                         ┌───────────────────────────────────────────────────────────┐
                         │                      Public Data Sources                  │
                         │  - PRJNA952168 SRA (16S & ITS)    - ESDAC LUCAS 2018      │
                         │  - PRJNA283936 (Mangrove 16S)     - Synthetic Mock Engine │
                         └─────────────────────────────┬─────────────────────────────┘
                                                       │
                                                       ▼
                         ┌───────────────────────────────────────────────────────────┐
                         │ Milestone 1: Data Ingestion, Auditing & Pairing (src/data)│
                         │  - BARCODE_ID deterministic pairing (16S + ITS)           │
                         │  - Geographic deduplication (881 unique sites)            │
                         │  - ESDAC edaphic & bioclimatic feature extraction         │
                         │  - Audit manifests, QC validation & Mock generator        │
                         └─────────────────────────────┬─────────────────────────────┘
                                                       │
                                                       ▼
                         ┌───────────────────────────────────────────────────────────┐
                         │ Milestone 2: Bioinformatics & Matrix Prep (src/bioinfo)   │
                         │  - FASTQ dual-orientation & primer matching (515F/ITS9mun)│
                         │  - ASV/OTU table ingestion (Illumina 16S & PacBio ITS)    │
                         │  - SILVA/UNITE taxonomic rollups (Phylum -> Genus)        │
                         │  - Read depth filtering & SRS normalization (40k / 502)   │
                         └─────────────────────────────┬─────────────────────────────┘
                                                       │
                                                       ▼
                         ┌───────────────────────────────────────────────────────────┐
                         │ Milestone 3: Compositional Ecology & PERMANOVA (src/eco)  │
                         │  - Alpha diversity (Shannon, Simpson, Richness, Chao1)    │
                         │  - CLR / robust CLR transformation for simplex data       │
                         │  - Beta diversity (Bray-Curtis, Aitchison distance)       │
                         │  - Permutational MANOVA + Multivariate Dispersion Tests   │
                         └─────────────────────────────┬─────────────────────────────┘
                                                       │
                                                       ▼
                         ┌───────────────────────────────────────────────────────────┐
                         │ Milestone 4: Spatial ML & Hypothesis Testing (src/models) │
                         │  - Zero-leakage Spatial CV (SpatialGroupKFold, Buffering) │
                         │  - Predictors: Set A (16S), Set B (ITS), Set C (Integr.)  │
                         │  - Supervised Classifiers (RF, GBDT, ElasticNet, SVM)     │
                         │  - Balanced Metrics, 1,000x Bootstrap CIs, Confusion Mat. │
                         │  - Paired Wilcoxon test of H0 (Multi-marker superiority)  │
                         └─────────────────────────────┬─────────────────────────────┘
                                                       │
                                                       ▼
                         ┌───────────────────────────────────────────────────────────┐
                         │ Milestone 5: CLI, Config & Pipeline Orchestration (main)  │
                         │  - YAML configuration with fixed seeds                    │
                         │  - Unified CLI entry points (ingest, eco, train, eval)    │
                         │  - Automated artifact export (JSON, CSV, figures)         │
                         └───────────────────────────────────────────────────────────┘
```

---

## Code Layout

```
Soil-microbiome-spatial/
├── config/
│   └── pipeline_config.yaml         # Pinned parameters, seeds, paths, thresholds
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── sra_metadata.py          # SRA/ENA metadata fetcher & schema parser
│   │   ├── esdac_parser.py          # ESDAC edaphic/climatic metadata parser
│   │   ├── pairing_engine.py        # Strict 1:1 biological pairing enforcer
│   │   ├── sample_manifest.py       # Audited paired sample manifest generator
│   │   └── mock_generator.py        # Deterministic synthetic mock dataset generator
│   ├── bioinformatics/
│   │   ├── __init__.py
│   │   ├── fastq_qc.py              # Primer trimming & dual-orientation resolution
│   │   ├── count_matrix.py          # ASV/OTU table parser and sparse/dense handler
│   │   ├── taxonomy.py              # SILVA/UNITE hierarchy parser & taxon rollup
│   │   └── normalization.py         # SRS normalization & depth threshold filter
│   ├── ecology/
│   │   ├── __init__.py
│   │   ├── alpha_diversity.py       # Richness, Shannon, Simpson, Chao1 indices
│   │   ├── compositional.py         # Centered Log-Ratio (CLR) & pseudocount models
│   │   ├── beta_diversity.py        # Bray-Curtis and Aitchison distance matrices
│   │   └── multivariate_stats.py    # Native PERMANOVA & betadisper dispersion tests
│   ├── spatial_cv/
│   │   ├── __init__.py
│   │   ├── spatial_partitioner.py   # SpatialGroupKFold, SpatialKMeans, Buffering
│   │   └── leakage_guard.py         # Mathematical assertions for zero spatial leakage
│   ├── models/
│   │   ├── __init__.py
│   │   ├── feature_builder.py       # Feature representations: Set A, Set B, Set C
│   │   ├── classifiers.py           # Random Forest, GBDT, ElasticNet, SVM wrappers
│   │   └── multi_marker_fusion.py   # Early fusion (concat) & Late fusion (stacking)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               # Balanced accuracy, Macro-F1, confusion matrix
│   │   ├── bootstrap_ci.py          # Stratified spatial bootstrap confidence intervals
│   │   └── hypothesis_test.py       # Paired Wilcoxon signed-rank test for H0
│   └── pipeline/
│       ├── __init__.py
│       ├── orchestrator.py          # Modular pipeline execution flow
│       └── cli.py                   # Subcommand CLI parser
├── main.py                          # Top-level CLI entry point
├── tests/
│   ├── conftest.py                  # Pytest fixtures and test mock data
│   ├── test_data_pairing.py         # R1 test: Pairing integrity & ESDAC joins
│   ├── test_bioinformatics.py       # R2 test: QC, SRS normalization, count matrices
│   ├── test_ecology.py              # R2 test: CLR, Aitchison, PERMANOVA, betadisper
│   ├── test_spatial_cv.py           # R3 test: Zero spatial leakage assertions
│   ├── test_models.py               # R3 test: Classifiers & multi-marker fusion
│   ├── test_evaluation.py           # R3 test: Metrics, bootstrap CIs, H0 tests
│   └── test_reproducibility.py      # R4 test: Fixed seed deterministic repeatability
├── requirements.txt                 # Pinned dependencies
└── pyproject.toml                   # Project metadata & build tool configuration
```

---

## Feature Inventory

Every feature discovered during the Survey phase is mapped to a designated Milestone:

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | SRA Metadata Parsing | Query and parse SRA/ENA run info for BioProject PRJNA952168 | M1 | Survey Miner & Data |
| 2 | ESDAC Edaphic Ingestion | Ingest ESDAC LUCAS 2018 soil properties (pH, OC, N, P, K, texture) | M1 | Survey Miner & Data |
| 3 | Strict Sample Pairing | Deterministic 1:1 pairing on integer `BARCODE_ID` / `BioSample` | M1 | ORIGINAL_REQUEST R1 |
| 4 | Re-sampled Site Deduplication | Detect and group duplicate sites (59/138, 102/601, 113/717, 116/258) | M1 | Survey Data |
| 5 | Paired Sample Manifest | Export structured manifest linking runs, barcodes, coordinates, classes | M1 | ORIGINAL_REQUEST R1 |
| 6 | PRJNA283936 Mangrove Ingestion | Isolated ingestion of complementary 16S-only mangrove dataset | M1 | ORIGINAL_REQUEST R1 |
| 7 | Synthetic Mock Generator | Deterministic generator of paired 16S/ITS profiles, metadata, coordinates | M1 | Survey Arch & Data |
| 8 | Variable Dictionary & Schema | JSON/YAML schema validating all edaphic and metadata fields | M1 | ORIGINAL_REQUEST R1 |
| 9 | 16S Dual-Orientation Handling | Detect and resolve mixed 5'->3' and 3'->5' reads from TruSeq adapters | M2 | Survey Data & Miner |
| 10 | Primer Matching & Trimming | Match 515F/926R (16S) and ITS9mun/ITS4ngsUni (ITS) primer regions | M2 | Survey Miner |
| 11 | ASV/OTU Count Table Ingestion | Dual-entrypoint parser for sparse/dense count matrices | M2 | ORIGINAL_REQUEST R2 |
| 12 | Taxonomic Annotation Parser | Parse SILVA (16S) and UNITE (ITS) taxonomic strings to rollups | M2 | ORIGINAL_REQUEST R2 |
| 13 | Read Depth Filtering | Filter low-abundance profiles below sequencing depth thresholds | M2 | Survey Miner |
| 14 | SRS Depth Normalization | Scale with Ranked Subsampling (16S: 40,109; ITS: 502 reads) | M2 | Survey Miner |
| 15 | Taxon Rollup Aggregator | Aggregate counts by Phylum, Class, Order, Family, Genus | M2 | ORIGINAL_REQUEST R2 |
| 16 | Alpha Diversity Metrics | Observed Richness, Shannon, Simpson, and Chao1 indices | M3 | ORIGINAL_REQUEST R2 |
| 17 | Simplex Pseudocount Imputation | Multiplicative replacement / robust pseudocounts for count zeros | M3 | Survey Miner & Arch |
| 18 | Centered Log-Ratio (CLR) | Compositional CLR and robust CLR transformations | M3 | Survey Miner & Arch |
| 19 | Bray-Curtis Dissimilarity | Pairwise Bray-Curtis distance matrix computation | M3 | ORIGINAL_REQUEST R2 |
| 20 | Aitchison Distance Matrix | Pairwise Euclidean distance on CLR-transformed compositions | M3 | ORIGINAL_REQUEST R2 |
| 21 | Native PERMANOVA Routine | Permutational MANOVA testing microbial association with soil/land cover | M3 | ORIGINAL_REQUEST R2 |
| 22 | Multivariate Dispersion Test | `betadisper` equivalent testing homogeneity of multivariate dispersions | M3 | ORIGINAL_REQUEST R2 |
| 23 | Spatial Distance Matrix | Compute pairwise haversine / Euclidean geographic distances | M4 | Survey Arch |
| 24 | Spatial Blocking Partitioner | SpatialGroupKFold and SpatialKMeans clustering for CV partitions | M4 | ORIGINAL_REQUEST R3 |
| 25 | Spatial Buffer Exclusion | Optional spatial buffer to exclude borderline spatial autocorrelation | M4 | Survey Miner |
| 26 | Class Representation Enforcer | Ensure all test classes are represented in corresponding training fold | M4 | Survey Miner |
| 27 | Zero Spatial Leakage Assertions| Programmatic verification that Sites(train) ∩ Sites(test) = ∅ | M4 | ORIGINAL_REQUEST R3 |
| 28 | Single-Marker Feature Sets | Feature builders for Set A (16S Bacteria) and Set B (ITS Fungi) | M4 | ORIGINAL_REQUEST R3 |
| 29 | Early Fusion Multi-Marker Set | Feature builder for Set C (Concatenated 16S + ITS) | M4 | ORIGINAL_REQUEST R3 |
| 30 | Late Fusion Multi-Marker Set | Ensemble stacking combining independent 16S and ITS predictions | M4 | Survey Miner |
| 31 | Supervised Classifiers | Scikit-learn Random Forest, GBDT, ElasticNet Logistic, and Linear SVM | M4 | ORIGINAL_REQUEST R3 |
| 32 | Balanced Accuracy & Macro-F1 | Imbalance-adjusted evaluation metrics per fold and aggregate | M4 | ORIGINAL_REQUEST R3 |
| 33 | Confusion Matrix Analytics | Per-class true/false positives and confusion matrices | M4 | ORIGINAL_REQUEST R3 |
| 34 | Spatial Bootstrap CIs | 1,000-iteration spatial bootstrap confidence intervals | M4 | ORIGINAL_REQUEST R3 |
| 35 | Null Hypothesis Tester | Paired Wilcoxon signed-rank test evaluating H0 (Set C <= max(A, B)) | M4 | Survey Miner |
| 36 | Pinned YAML Configuration | Central config file with fixed seeds, paths, hyperparameters | M5 | ORIGINAL_REQUEST R4 |
| 37 | Unified CLI Entry Point | `main.py` CLI supporting subcommands (ingest, eco, train, eval, all) | M5 | Survey Arch |
| 38 | Automated Artifact Exporter | Structured output directories exporting CSV summaries, JSON metrics | M5 | ORIGINAL_REQUEST R4 |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Data Ingestion, Auditing & Pairing | SRA parsing, ESDAC metadata, strict 1:1 pairing, mock generator, manifest export | none | COMPLETED |
| M2 | Bioinformatics & Matrix Processing | Dual-orientation QC, ASV ingestion, SILVA/UNITE taxonomy, SRS normalization | M1 | COMPLETED |
| M3 | Compositional Ecology & PERMANOVA | Alpha diversity, CLR transforms, Bray-Curtis & Aitchison beta distances, PERMANOVA & betadisper | M2 | COMPLETED |
| M4 | Spatial ML & Null Hypothesis Testing| Zero-leakage Spatial CV, Set A/B/C feature sets, Classifiers, Balanced metrics, Bootstrap CIs, H0 test | M1, M3 | COMPLETED |
| M5 | CLI, Configuration & Automation | Config YAML with seeds, top-level CLI, end-to-end integration, artifact outputs | M1, M2, M3, M4 | COMPLETED |
| M-E2E| E2E Testing Suite Track | Opaque-box test suite across Tiers 1-4, test runner, TEST_READY.md | Independent | COMPLETED |
| M-FINAL| Final Verification & Hardening | Pass 100% of E2E test suite (325/325 passing tests, full CLI verification) | M1-M5, M-E2E | COMPLETED |

---

## Interface Contracts

### M1 (Data) ↔ M2 (Bioinformatics)
- **Data file**: `data/processed/paired_sample_manifest.csv`
  - Columns: `sample_id` (str), `barcode_id` (int), `site_id` (str), `latitude` (float), `longitude` (float), `land_cover` (str), `land_cover_code` (int), `sra_16s_run` (str), `sra_its_run` (str), `esdac_ph` (float), `esdac_oc` (float), `esdac_n` (float), `esdac_clay` (float), `esdac_silt` (float), `esdac_sand` (float).
- **Function**: `load_paired_manifest(filepath: str) -> pd.DataFrame`

### M2 (Bioinformatics) ↔ M3 (Ecology)
- **Data files**: 
  - `data/processed/bacteria_16s_asv_matrix.csv` (samples x ASVs)
  - `data/processed/fungi_its_asv_matrix.csv` (samples x ASVs)
  - `data/processed/taxonomy_16s.csv`, `data/processed/taxonomy_its.csv`
- **Functions**:
  - `load_count_matrix(filepath: str) -> np.ndarray, list[str], list[str]` (counts, sample_ids, feature_ids)
  - `normalize_srs(counts: np.ndarray, target_depth: int, seed: int = 42) -> np.ndarray`

### M3 (Ecology) ↔ M4 (Spatial ML)
- **Data files**:
  - `data/processed/clr_bacteria_16s.csv`
  - `data/processed/clr_fungi_its.csv`
  - `data/processed/alpha_diversity.csv` (Shannon, Simpson, Richness per marker)
- **Functions**:
  - `clr_transform(matrix: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray`
  - `run_permanova(distance_matrix: np.ndarray, groupings: np.ndarray, permutations: int = 999, seed: int = 42) -> dict`
  - `run_betadisper(distance_matrix: np.ndarray, groupings: np.ndarray) -> dict`

### M4 (Spatial ML) ↔ M5 (CLI & Orchestration)
- **Functions**:
  - `create_spatial_splits(manifest: pd.DataFrame, n_splits: int, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]`
  - `train_and_evaluate_models(X_dict: dict, y: np.ndarray, splits: list, config: dict) -> dict`
  - `test_multimarker_null_hypothesis(scores_a: np.ndarray, scores_b: np.ndarray, scores_c: np.ndarray) -> dict`
- **Output files**:
  - `results/spatial_cv_metrics.json`
  - `results/confusion_matrices.json`
  - `results/hypothesis_test_results.json`
