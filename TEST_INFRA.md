# TEST_INFRA — Comprehensive Test Architecture & Infrastructure

**Project**: Soil Microbiome Environmental Discrimination Pipeline under Spatial Cross-Validation  
**Document**: Test Infrastructure Specification & Execution Protocol  
**Author**: E2E Test Suite Designer & Writer (`e2e_test_writer_1`)  
**Target Environment**: Windows host / Python 3.12+ / Pytest 8+  
**Date**: 2026-10-29  

---

## 1. Test Philosophy & Design Principles

The testing framework for the Soil Microbiome Environmental Discrimination Pipeline is architected to guarantee scientific validity, computational reproducibility, and strict adherence to domain invariants. The framework adheres to four core principles:

### 1.1 Opaque-Box & Requirement-Driven Verification
Tests treat system components as opaque units evaluated against published interfaces, schema contracts, and domain specifications documented in `ORIGINAL_REQUEST.md`, `PROJECT.md`, and authoritative peer-reviewed literature (Labouyrie et al. *Nat Commun* 2023; Gloor et al. *Front Microbiol* 2017; Roberts et al. *Ecography* 2017). Tests assert *what* the system must compute, rather than *how* private functions are structured internally.

### 1.2 Mathematical & Ecological Invariants
Biological amplicon sequencing data and spatial ecological coordinates follow strict mathematical laws that the test suite enforces as non-negotiable invariants:
1. **Compositional Simplex Invariants**: Microbiome abundance data are non-negative vectors subject to arbitrary sequencing depth constraints. Centered Log-Ratio (CLR) transformed compositions must strictly sum to zero ($\sum_{j=1}^D \text{clr}(x)_j = 0$), preserve scale invariance ($\text{clr}(c \cdot x) = \text{clr}(x)$ for any scalar $c > 0$), and maintain sub-compositional coherence.
2. **Metric Axioms of Aitchison Distance**: Pairwise Aitchison distances must satisfy non-negativity ($d(x, y) \ge 0$), identity of indiscernibles ($d(x, y) = 0 \iff x = c \cdot y$), symmetry ($d(x, y) = d(y, x)$), and the triangle inequality ($d(x, z) \le d(x, y) + d(y, z)$).
3. **SRS Depth Normalization**: Ranked Subsampling must output integer count vectors whose column sums equal exactly the target sequencing depth ($C_{min} = 40,109$ for 16S; $502$ for ITS) without altering relative rank ordering of prevalent taxa.
4. **PERMANOVA & Betadisper Dispersions**: Pseudo-$F$ test statistics must be non-negative, and $R^2$ effect sizes must lie strictly in $[0, 1]$. Group dispersion tests must flag multivariate heteroscedasticity that could confound centroid discrimination.

### 1.3 Zero Spatial Data Leakage Guarantee
Soil microbiome assemblages are spatially autocorrelated. Standard random cross-validation allows geographic replicates and nearby sites to leak between training and testing partitions, artificially inflating generalization accuracy. The test suite enforces:
1. **Index Disjointness**: $\text{Train}_{\text{idx}} \cap \text{Test}_{\text{idx}} = \emptyset$.
2. **Site ID Disjointness**: $\text{Sites}(\text{Train}) \cap \text{Sites}(\text{Test}) = \emptyset$. Replicates and re-sampled sites (e.g. LUCAS pairs 59/138, 102/601, 113/717, 116/258) must strictly reside within the same validation fold.
3. **Exact Coordinate Disjointness**: No spatial coordinate $(\text{lat}, \text{lon})$ in a test set may occur in the corresponding training set.
4. **Buffer Zone Isolation**: When spatial buffer $R_{buffer} > 0$ is specified, $\min_{\mathbf{u} \in \text{Train}, \mathbf{v} \in \text{Test}} \text{Haversine}(\mathbf{u}, \mathbf{v}) \ge R_{buffer}$.
5. **Class Representation Invariant**: Every target environmental category in a spatial test fold must have representation in the corresponding training fold ($Y_{\text{test}} \subseteq Y_{\text{train}}$) to ensure fair multi-class classification.

### 1.4 Strict Multi-Marker Hypothesis Testing ($H_0$)
The central scientific question evaluates whether multi-marker integration (Set C: 16S + ITS) outperforms the single best-performing individual marker:
$$H_0: \text{Score}_{\text{Set C}} - \max\left(\text{Score}_{\text{Set A}}, \text{Score}_{\text{Set B}}\right) \le 0$$
The test framework asserts that if the empirical gain $\Delta \le 0$ or the paired Wilcoxon signed-rank test yields $p \ge 0.05$, the pipeline transparently accepts $H_0$ and documents non-superiority, rather than producing false positive claims.

---

## 2. Four-Tier Test Architecture

The testing hierarchy organizes test coverage into four progressive tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│               Tier 4: Real-World E2E Scenarios (tests/)                │
│  - 715-site continental LUCAS workflow    - Null hypothesis testing    │
│  - Spatial leave-cluster-out validation   - Multi-marker comparison    │
│  - Brazilian mangrove isolated workflow   - End-to-end artifact export │
├────────────────────────────────────────────────────────────────────────┤
│             Tier 3: Pairwise Cross-Feature Integration (tests/)        │
│  - Pairing engine + ESDAC metadata join   - CLR + Aitchison distance   │
│  - SpatialGroupKFold + Zero Leakage + RF  - Early vs Late fusion fit   │
│  - SRS normalization + Alpha/Beta stats   - Train-fold edaphic impute  │
├────────────────────────────────────────────────────────────────────────┤
│                Tier 2: Boundary, Adversarial & Robustness (tests/)     │
│  - Single-sample folds & extreme skew     - Zeroes in simplex vectors  │
│  - Re-sampled spatial duplicate pairs     - Sub-threshold read depths  │
│  - Corrupt SRA tables & missing edaphic   - Dual-orientation inversion │
├────────────────────────────────────────────────────────────────────────┤
│             Tier 1: Feature Unit & Interface Contracts (tests/)        │
│  - Individual feature coverage (>=5 test cases per feature #1..#38)    │
│  - Input/output typing, return contracts, schema validations           │
│  - Deterministic random seed propagation across NumPy/SciPy/Sklearn    │
└────────────────────────────────────────────────────────────────────────┘
```

### Tier 1: Feature Unit & Interface Contract Tests (`tests/test_tier1_features.py`)
- Focuses on individual features across Milestones M1, M2, M3, M4, and M5.
- Validates that every public function and component adheres to the interface contracts defined in `PROJECT.md § Interface Contracts`.
- Enforces at least 5 distinct test cases per feature:
  - Happy path / standard input.
  - Input type and shape validation.
  - Mathematical correctness against analytical reference oracles.
  - Empty or minimal input handling.
  - Configurable parameter validation.

### Tier 2: Boundary, Adversarial & Robustness Tests (`tests/test_tier2_boundaries.py`)
- Subjecting data parsers, ecological algorithms, and spatial ML models to edge cases, pathological inputs, and extreme boundary conditions.
- Tests include:
  - Sparse simplex vectors containing 99.9% zeros.
  - Library read depths exactly at the boundary ($C = 40,109$ and $C = 40,108$ for 16S; $C = 502$ and $C = 501$ for ITS).
  - Spatial duplicate locations (identical coordinates across different sample IDs).
  - Extreme class imbalance (e.g. 18 samples in Intensive Grassland vs 264 in Non-permanent Crop).
  - Boundary coordinates ($\text{Lat} = \pm 90^\circ, \text{Lon} = \pm 180^\circ$, antimeridian crossing).
  - Corrupt or malformed FASTA/FASTQ headers and TruSeq inverted orientations.

### Tier 3: Pairwise Cross-Feature Integration Tests (`tests/test_tier3_pairwise.py`)
- Exercises the interplay between pairs and triplets of features operating across module boundaries.
- Verifies that intermediate data pipelines do not introduce subtle information leaks or formatting corruptions.
- Core combinations:
  - Strict 1:1 Biological Pairing Engine + ESDAC Physicochemical Ingestion.
  - SRS Normalization + Simplex Zero Imputation + CLR Transformation.
  - CLR Transformation + Aitchison Distance Matrix + Native PERMANOVA Routine.
  - SpatialGroupKFold Partitioner + Zero Spatial Leakage Guard + Supervised Classifier Fit.
  - Early Fusion Feature Builder + Late Fusion Meta-Stacking Ensemble.
  - Inside-Fold Preprocessing Isolation (StandardScaler / Imputer fitted strictly on train fold).

### Tier 4: Real-World E2E Scenario Workflows (`tests/test_tier4_scenarios.py`)
- Emulates the end-to-end scientific workflow as executed in research practice.
- Tests:
  - Complete 715-site European LUCAS soil study workflow from raw metadata to final report.
  - Comparative benchmarking of Set A (16S Bacteria) vs Set B (ITS Fungi) vs Set C (Integrated Multi-Marker).
  - Paired Wilcoxon Signed-Rank Test evaluating Null Hypothesis $H_0$.
  - 1,000-iteration Spatial Bootstrap Confidence Intervals for balanced accuracy and macro-F1.
  - Completely isolated execution of Brazilian semi-arid mangrove benchmark (PRJNA283936).
  - Pipeline artifact generation, JSON metric serializability, and seed determinism across runs.

---

## 3. Feature Inventory Mapping to Test Hierarchy

All 38 features enumerated in `PROJECT.md § Feature Inventory` are systematically mapped to specific test modules, classes, and tiers:

| # | Feature Name | Milestone | Primary Test Module | Primary Test Class / Function Prefix | Target Tier |
|---|--------------|-----------|---------------------|---------------------------------------|-------------|
| 1 | SRA Metadata Parsing | M1 | `tests/test_tier1_features.py` | `TestFeature01_SRAMetadataParsing` | Tier 1, 2 |
| 2 | ESDAC Edaphic Ingestion | M1 | `tests/test_tier1_features.py` | `TestFeature02_ESDACEdaphicIngestion` | Tier 1, 2 |
| 3 | Strict Sample Pairing | M1 | `tests/test_tier1_features.py` | `TestFeature03_StrictSamplePairing` | Tier 1, 2, 3 |
| 4 | Re-sampled Site Deduplication | M1 | `tests/test_tier1_features.py` | `TestFeature04_ResampledSiteDeduplication` | Tier 1, 2, 3 |
| 5 | Paired Sample Manifest | M1 | `tests/test_tier1_features.py` | `TestFeature05_PairedSampleManifest` | Tier 1, 3 |
| 6 | PRJNA283936 Mangrove Ingestion | M1 | `tests/test_tier1_features.py` | `TestFeature06_MangroveIngestion` | Tier 1, 4 |
| 7 | Synthetic Mock Generator | M1 | `tests/test_tier1_features.py` | `TestFeature07_SyntheticMockGenerator` | Tier 1, 4 |
| 8 | Variable Dictionary & Schema | M1 | `tests/test_tier1_features.py` | `TestFeature08_VariableDictionarySchema` | Tier 1, 2 |
| 9 | 16S Dual-Orientation Handling | M2 | `tests/test_tier1_features.py` | `TestFeature09_DualOrientationHandling` | Tier 1, 2 |
| 10 | Primer Matching & Trimming | M2 | `tests/test_tier1_features.py` | `TestFeature10_PrimerMatchingTrimming` | Tier 1, 2 |
| 11 | ASV/OTU Count Table Ingestion | M2 | `tests/test_tier1_features.py` | `TestFeature11_CountTableIngestion` | Tier 1, 2 |
| 12 | Taxonomic Annotation Parser | M2 | `tests/test_tier1_features.py` | `TestFeature12_TaxonomicAnnotationParser` | Tier 1, 2 |
| 13 | Read Depth Filtering | M2 | `tests/test_tier1_features.py` | `TestFeature13_ReadDepthFiltering` | Tier 1, 2 |
| 14 | SRS Depth Normalization | M2 | `tests/test_tier1_features.py` | `TestFeature14_SRSNormalization` | Tier 1, 2, 3 |
| 15 | Taxon Rollup Aggregator | M2 | `tests/test_tier1_features.py` | `TestFeature15_TaxonRollupAggregator` | Tier 1, 3 |
| 16 | Alpha Diversity Metrics | M3 | `tests/test_tier1_features.py` | `TestFeature16_AlphaDiversityMetrics` | Tier 1, 3 |
| 17 | Simplex Pseudocount Imputation | M3 | `tests/test_tier1_features.py` | `TestFeature17_PseudocountImputation` | Tier 1, 2 |
| 18 | Centered Log-Ratio (CLR) | M3 | `tests/test_tier1_features.py` | `TestFeature18_CenteredLogRatio` | Tier 1, 2, 3 |
| 19 | Bray-Curtis Dissimilarity | M3 | `tests/test_tier1_features.py` | `TestFeature19_BrayCurtisDissimilarity` | Tier 1, 2 |
| 20 | Aitchison Distance Matrix | M3 | `tests/test_tier1_features.py` | `TestFeature20_AitchisonDistanceMatrix` | Tier 1, 2, 3 |
| 21 | Native PERMANOVA Routine | M3 | `tests/test_tier1_features.py` | `TestFeature21_NativePERMANOVA` | Tier 1, 3 |
| 22 | Multivariate Dispersion Test | M3 | `tests/test_tier1_features.py` | `TestFeature22_MultivariateDispersion` | Tier 1, 3 |
| 23 | Spatial Distance Matrix | M4 | `tests/test_tier1_features.py` | `TestFeature23_SpatialDistanceMatrix` | Tier 1, 2 |
| 24 | Spatial Blocking Partitioner | M4 | `tests/test_tier1_features.py` | `TestFeature24_SpatialBlockingPartitioner` | Tier 1, 3, 4 |
| 25 | Spatial Buffer Exclusion | M4 | `tests/test_tier1_features.py` | `TestFeature25_SpatialBufferExclusion` | Tier 1, 2 |
| 26 | Class Representation Enforcer | M4 | `tests/test_tier1_features.py` | `TestFeature26_ClassRepresentationEnforcer` | Tier 1, 2, 3 |
| 27 | Zero Spatial Leakage Assertions | M4 | `tests/test_tier1_features.py` | `TestFeature27_ZeroSpatialLeakage` | Tier 1, 3, 4 |
| 28 | Single-Marker Feature Sets | M4 | `tests/test_tier1_features.py` | `TestFeature28_SingleMarkerFeatureSets` | Tier 1, 3, 4 |
| 29 | Early Fusion Multi-Marker Set | M4 | `tests/test_tier1_features.py` | `TestFeature29_EarlyFusionSet` | Tier 1, 3, 4 |
| 30 | Late Fusion Multi-Marker Set | M4 | `tests/test_tier1_features.py` | `TestFeature30_LateFusionSet` | Tier 1, 3, 4 |
| 31 | Supervised Classifiers | M4 | `tests/test_tier1_features.py` | `TestFeature31_SupervisedClassifiers` | Tier 1, 3 |
| 32 | Balanced Accuracy & Macro-F1 | M4 | `tests/test_tier1_features.py` | `TestFeature32_BalancedMetrics` | Tier 1, 2, 3 |
| 33 | Confusion Matrix Analytics | M4 | `tests/test_tier1_features.py` | `TestFeature33_ConfusionMatrixAnalytics` | Tier 1, 2 |
| 34 | Spatial Bootstrap CIs | M4 | `tests/test_tier1_features.py` | `TestFeature34_SpatialBootstrapCIs` | Tier 1, 4 |
| 35 | Null Hypothesis Tester (H0) | M4 | `tests/test_tier1_features.py` | `TestFeature35_NullHypothesisTester` | Tier 1, 4 |
| 36 | Pinned YAML Configuration | M5 | `tests/test_tier1_features.py` | `TestFeature36_PinnedYAMLConfiguration` | Tier 1, 2 |
| 37 | Unified CLI Entry Point | M5 | `tests/test_tier1_features.py` | `TestFeature37_UnifiedCLIEntryPoint` | Tier 1, 4 |
| 38 | Automated Artifact Exporter | M5 | `tests/test_tier1_features.py` | `TestFeature38_AutomatedArtifactExporter` | Tier 1, 4 |

---

## 4. Test Infrastructure & Fixture Architecture (`tests/conftest.py`)

The test infrastructure provides a self-contained test environment implemented in `tests/conftest.py`:

### 4.1 Mock Synthetic Data Generators
1. **`mock_paired_survey_data`**:
   - Generates deterministic tabular metadata matching the 885-sample / 715-core LUCAS layout.
   - Includes integer `barcode_id`, `site_id`, coordinates (`latitude`, `longitude`), `land_cover` (`Cropland`, `Grassland`, `Woodland`), and physicochemical variables (`pH`, `OC`, `N`, `clay`, `silt`, `sand`).
   - Includes the 4 exact spatial re-sampling pairs (59/138, 102/601, 113/717, 116/258).
2. **`mock_count_matrices`**:
   - Deterministic 16S bacterial and ITS fungal count matrices across paired samples.
   - Realistic library depths (16S: 40k–150k reads; ITS: 500–10k reads).
   - Known zero-sparsity and non-negative integer counts.
3. **`mock_spatial_coordinates`**:
   - European bounding box coordinates (Latitude $35^\circ\text{N}$ to $70^\circ\text{N}$, Longitude $10^\circ\text{W}$ to $35^\circ\text{E}$).
   - Brazilian mangrove coordinates (Latitude $-3.5^\circ$ to $-4.5^\circ$, Longitude $-38^\circ$ to $-40^\circ$).

### 4.2 Analytical Oracles & Reference Implementations
The test infrastructure embeds pure Python reference oracles for zero-dependency mathematical verification:
- `oracle_clr(counts, pseudocount)`: Computes exact geometric mean and Centered Log-Ratio transform.
- `oracle_aitchison_distance(x, y)`: Computes analytical Euclidean distance on CLR vectors.
- `oracle_shannon_entropy(p)`: Computes analytical $-\sum p \ln p$.
- `oracle_bray_curtis(u, v)`: Computes analytical $\frac{\sum |u_i - v_i|}{\sum (u_i + v_i)}$.
- `oracle_srs(counts, depth, seed)`: Exact scaling with ranked subsampling.
- `oracle_haversine(lat1, lon1, lat2, lon2)`: Exact great-circle geographic distance on sphere ($R = 6,371\text{ km}$).
- `oracle_wilcoxon_signed_rank(x, y)`: Analytical Wilcoxon test for paired differences.

---

## 5. Test Runner & Invocation Protocol

### 5.1 Standard Pytest Invocation
To execute the complete test suite from the repository root:

```bash
# Run all tests with verbose output
pytest -v tests/

# Run specific tier
pytest -v tests/test_tier1_features.py
pytest -v tests/test_tier2_boundaries.py
pytest -v tests/test_tier3_pairwise.py
pytest -v tests/test_tier4_scenarios.py

# Run with test coverage
pytest --cov=src tests/

# Run with fail-fast on first failure
pytest -x -v tests/
```

### 5.2 Deterministic Execution Guarantee
Every test module sets and asserts fixed random seeds (`seed = 42`) across Python `random`, `numpy.random`, and scikit-learn estimators. Two independent executions of the test suite produce identical numerical test results down to machine floating point precision.
