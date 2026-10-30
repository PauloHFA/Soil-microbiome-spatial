"""
conftest.py - Pytest fixtures and mock data infrastructure for Soil Microbiome Pipeline tests.
Provides synthetic datasets, mathematical oracles, and test environment utilities.
"""

import os
import math
import random
import tempfile
from typing import Dict, List, Tuple, Any
import numpy as np
import pytest

# Set deterministic seeds for test isolation
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ============================================================================
# Pure Python Analytical Oracles (Ground Truth Implementations)
# ============================================================================

def oracle_clr(counts: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
    """
    Authoritative reference oracle for Centered Log-Ratio (CLR) transformation.
    x_clr_i = ln(x_i / g(x)) where g(x) is geometric mean.
    Sum of CLR values across features must be identically zero.
    """
    matrix = np.array(counts, dtype=np.float64)
    # Add pseudocount for simplex zero replacement
    matrix_pseudo = np.where(matrix <= 0, pseudocount, matrix)
    log_matrix = np.log(matrix_pseudo)
    geom_mean = np.mean(log_matrix, axis=-1, keepdims=True)
    return log_matrix - geom_mean


def oracle_aitchison_distance(x: np.ndarray, y: np.ndarray, pseudocount: float = 1e-6) -> float:
    """
    Authoritative reference oracle for Aitchison distance: Euclidean distance on CLR vectors.
    """
    clr_x = oracle_clr(np.array(x, dtype=np.float64), pseudocount=pseudocount)
    clr_y = oracle_clr(np.array(y, dtype=np.float64), pseudocount=pseudocount)
    return float(np.linalg.norm(clr_x - clr_y))


def oracle_bray_curtis(u: np.ndarray, v: np.ndarray) -> float:
    """
    Authoritative reference oracle for Bray-Curtis dissimilarity:
    BC = sum(|u_i - v_i|) / sum(u_i + v_i)
    """
    u = np.array(u, dtype=np.float64)
    v = np.array(v, dtype=np.float64)
    numerator = np.sum(np.abs(u - v))
    denominator = np.sum(u + v)
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def oracle_shannon_entropy(counts: np.ndarray) -> float:
    """
    Authoritative reference oracle for Shannon diversity index: H' = -sum(p_i * ln(p_i))
    """
    c = np.array(counts, dtype=np.float64)
    total = np.sum(c)
    if total <= 0:
        return 0.0
    p = c[c > 0] / total
    return float(-np.sum(p * np.log(p)))


def oracle_simpson_index(counts: np.ndarray) -> float:
    """
    Authoritative reference oracle for Gini-Simpson index: 1 - sum(p_i^2)
    """
    c = np.array(counts, dtype=np.float64)
    total = np.sum(c)
    if total <= 0:
        return 0.0
    p = c / total
    return float(1.0 - np.sum(p ** 2))


def oracle_chao1(counts: np.ndarray) -> float:
    """
    Authoritative reference oracle for Chao1 richness estimator:
    S_Chao1 = S_obs + (f1 * (f1 - 1)) / (2 * (f2 + 1)) if f2 > 0 else S_obs + f1 * (f1 - 1) / 2
    where f1 = singletons, f2 = doubletons.
    """
    c = np.array(counts, dtype=np.int64)
    s_obs = np.sum(c > 0)
    f1 = np.sum(c == 1)
    f2 = np.sum(c == 2)
    if f2 > 0:
        return float(s_obs + (f1 * (f1 - 1)) / (2.0 * (f2 + 1)))
    else:
        return float(s_obs + (f1 * (f1 - 1)) / 2.0)


def oracle_srs_normalization(counts: np.ndarray, target_depth: int, seed: int = 42) -> np.ndarray:
    """
    Authoritative reference oracle for Scaling with Ranked Subsampling (SRS).
    Scales counts to target_depth, sorts decimals, and allocates remaining counts.
    """
    c = np.array(counts, dtype=np.float64)
    total = np.sum(c)
    if total < target_depth:
        raise ValueError(f"Sample depth {total} below SRS target depth {target_depth}")
    
    scaled = (c / total) * target_depth
    floored = np.floor(scaled).astype(np.int64)
    remainder = int(target_depth - np.sum(floored))
    
    if remainder > 0:
        decimals = scaled - floored
        rng = np.random.RandomState(seed)
        # Add tiny jitter to break ties deterministically
        jittered = decimals + rng.uniform(0, 1e-9, size=len(decimals))
        top_indices = np.argsort(jittered)[::-1][:remainder]
        floored[top_indices] += 1
        
    return floored


def oracle_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle distance (km) between two points on Earth using Haversine formula.
    Radius = 6,371.0088 km.
    """
    r = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def oracle_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes macro-average recall across all unique classes present in y_true.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        mask = (y_true == c)
        total_c = np.sum(mask)
        if total_c == 0:
            continue
        tp = np.sum((y_pred == c) & mask)
        recalls.append(tp / total_c)
    return float(np.mean(recalls)) if recalls else 0.0


def oracle_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes macro-average F1-score across all unique classes in y_true.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def oracle_wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Pure Python implementation of paired Wilcoxon signed-rank test.
    Returns (statistic, p_value approximation).
    """
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    d = x - y
    d = d[d != 0]  # Remove zero differences
    n = len(d)
    if n == 0:
        return 0.0, 1.0
    
    ranks = np.argsort(np.abs(d)) + 1
    w_plus = np.sum(ranks[d > 0])
    w_minus = np.sum(ranks[d < 0])
    w = min(w_plus, w_minus)
    
    # Asymptotic normal approximation for n >= 10
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    z = (w - mean_w) / math.sqrt(var_w) if var_w > 0 else 0.0
    # Standard normal two-tailed p-value approximation
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return float(w), float(p_val)


# ============================================================================
# Synthetic Mock Datasets and Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def sample_size() -> int:
    """Standard cohort size for integration testing: 60 paired samples."""
    return 60


@pytest.fixture(scope="session")
def mock_lucas_manifest(sample_size) -> Dict[str, Any]:
    """
    Generates a structured mock LUCAS manifest conforming to M1 requirements:
    Includes 60 paired biological samples across 4 biogeographical regions,
    3 macro land cover classes, 6 fine classes, edaphic ISO variables,
    and 4 spatial duplicate pairs (59/138, 102/601, 113/717, 116/258).
    """
    rng = np.random.RandomState(SEED)
    
    macro_classes = ["Cropland", "Grassland", "Woodland"]
    fine_classes = [
        "Non-permanent crops", "Permanent crops", 
        "Intensive grassland", "Extensive grassland", 
        "Broadleaved forests", "Coniferous forests"
    ]
    regions = ["Atlantic", "Boreal", "Continental", "Mediterranean"]
    
    records = []
    # Seed 4 explicit duplicate pairs
    dup_pairs = [(59, 138), (102, 601), (113, 717), (116, 258)]
    dup_canonical = {
        59: 59, 138: 59,
        102: 102, 601: 102,
        113: 113, 717: 113,
        116: 116, 258: 116,
    }
    dup_coords = {
        59: (50.12, 4.35), 138: (50.12, 4.35),
        102: (44.50, 11.20), 601: (44.50, 11.20),
        113: (59.33, 18.06), 717: (59.33, 18.06),
        116: (37.98, 23.72), 258: (37.98, 23.72)
    }
    dup_region_map = {
        59: "Atlantic", 138: "Atlantic",
        102: "Mediterranean", 601: "Mediterranean",
        113: "Boreal", 717: "Boreal",
        116: "Mediterranean", 258: "Mediterranean"
    }
    
    barcodes = [59, 138, 102, 601, 113, 717, 116, 258]
    while len(barcodes) < sample_size:
        bc = int(rng.randint(1, 885))
        if bc not in barcodes:
            barcodes.append(bc)
            
    for idx, bc in enumerate(barcodes):
        if bc in dup_coords:
            lat, lon = dup_coords[bc]
            site_id = f"SITE_DUP_{dup_canonical[bc]}"
        else:
            lat = float(rng.uniform(36.0, 65.0))
            lon = float(rng.uniform(-8.0, 28.0))
            site_id = f"SITE_{bc:04d}"
            
        macro = macro_classes[idx % len(macro_classes)]
        fine = fine_classes[idx % len(fine_classes)]
        if bc in dup_region_map:
            region = dup_region_map[bc]
        else:
            region = regions[idx % len(regions)]
        
        raw_clay = float(rng.uniform(10.0, 40.0))
        raw_silt = float(rng.uniform(20.0, 50.0))
        raw_sand = 100.0 - raw_clay - raw_silt
        
        record = {
            "sample_id": f"LUCAS_{bc}",
            "barcode_id": bc,
            "site_id": site_id,
            "point_id": 100000 + bc,
            "latitude": lat,
            "longitude": lon,
            "biogeographical_region": region,
            "macro_land_cover": macro,
            "fine_land_cover": fine,
            "land_cover_code": idx % 6,
            "sra_16s_run": f"SRR2406{bc:04d}",
            "sra_its_run": f"SRR2409{bc:04d}",
            "reads_16s": int(rng.randint(45000, 120000)),
            "reads_its": int(rng.randint(600, 8000)),
            "ph_h2o": float(round(rng.uniform(4.0, 8.2), 2)),
            "organic_carbon": float(round(rng.uniform(8.0, 95.0), 1)),
            "nitrogen": float(round(rng.uniform(0.8, 6.5), 2)),
            "clay": float(round(raw_clay, 1)),
            "silt": float(round(raw_silt, 1)),
            "sand": float(round(raw_sand, 1)),
            "bulk_density": float(round(rng.uniform(0.9, 1.6), 2)),
            "is_paired": True
        }
        records.append(record)
        
    return {
        "records": records,
        "n_samples": len(records),
        "barcodes": barcodes,
        "duplicate_pairs": dup_pairs
    }


@pytest.fixture(scope="session")
def mock_count_data(mock_lucas_manifest) -> Dict[str, Any]:
    """
    Generates paired 16S and ITS count matrices:
    - 16S: 60 samples x 120 ASVs
    - ITS: 60 samples x 80 OTUs
    Counts follow negative binomial / log-normal abundance distribution with known zero sparsity.
    """
    rng = np.random.RandomState(SEED)
    n_samples = mock_lucas_manifest["n_samples"]
    n_asv_16s = 120
    n_otu_its = 80
    
    # Generate 16S counts (high depth ~ 40k - 100k)
    base_16s = rng.lognormal(mean=3.0, sigma=1.5, size=(n_samples, n_asv_16s))
    mask_16s = rng.uniform(0, 1, size=(n_samples, n_asv_16s)) > 0.45  # Sparsity ~45%
    base_16s[mask_16s] = 0.0
    scale_16s = rng.uniform(45000, 90000, size=(n_samples, 1)) / (np.sum(base_16s, axis=1, keepdims=True) + 1e-9)
    counts_16s = np.round(base_16s * scale_16s).astype(np.int64)
    
    # Generate ITS counts (lower depth ~ 600 - 5000)
    base_its = rng.lognormal(mean=2.0, sigma=1.2, size=(n_samples, n_otu_its))
    mask_its = rng.uniform(0, 1, size=(n_samples, n_otu_its)) > 0.55   # Sparsity ~55%
    base_its[mask_its] = 0.0
    scale_its = rng.uniform(600, 4000, size=(n_samples, 1)) / (np.sum(base_its, axis=1, keepdims=True) + 1e-9)
    counts_its = np.round(base_its * scale_its).astype(np.int64)
    
    feature_ids_16s = [f"ASV_16S_{i:04d}" for i in range(n_asv_16s)]
    feature_ids_its = [f"OTU_ITS_{i:04d}" for i in range(n_otu_its)]
    sample_ids = [r["sample_id"] for r in mock_lucas_manifest["records"]]
    
    return {
        "counts_16s": counts_16s,
        "counts_its": counts_its,
        "features_16s": feature_ids_16s,
        "features_its": feature_ids_its,
        "sample_ids": sample_ids,
        "n_samples": n_samples
    }


@pytest.fixture(scope="session")
def mock_taxonomy() -> Dict[str, Any]:
    """Taxonomic hierarchical strings conforming to SILVA (16S) and UNITE (ITS)."""
    phyla_16s = ["p__Acidobacteriota", "p__Proteobacteria", "p__Actinobacteriota", "p__Bacteroidota", "p__Firmicutes"]
    phyla_its = ["p__Ascomycota", "p__Basidiomycota", "p__Mortierellomycota", "p__Mucoromycota", "p__Glomeromycota"]
    
    tax_16s = [
        f"k__Bacteria; {phyla_16s[i % len(phyla_16s)]}; c__Class_{i % 8}; o__Order_{i % 15}; f__Fam_{i % 25}; g__Genus_{i % 40}; s__Species_{i}"
        for i in range(120)
    ]
    tax_its = [
        f"k__Fungi; {phyla_its[i % len(phyla_its)]}; c__Class_{i % 6}; o__Order_{i % 12}; f__Fam_{i % 20}; g__Genus_{i % 30}; s__Species_{i}"
        for i in range(80)
    ]
    return {"taxonomy_16s": tax_16s, "taxonomy_its": tax_its}


@pytest.fixture(scope="session")
def mock_mangrove_data() -> Dict[str, Any]:
    """
    Mock dataset for PRJNA283936 (Brazilian Mangrove Microbiomes).
    105 samples from 7 estuaries, bacterial 16S only (V3-V4).
    """
    rng = np.random.RandomState(SEED + 10)
    n_samples = 105
    n_asv = 50
    estuaries = ["Ceara", "Cocó", "Pacoti", "Choró", "Pirangi", "Jaguaribe", "Acu"]
    
    records = []
    for i in range(n_samples):
        records.append({
            "sample_id": f"MANGROVE_{i+1:03d}",
            "bioproject": "PRJNA283936",
            "estuary": estuaries[i % len(estuaries)],
            "latitude": -3.7 - 0.1 * (i % 7),
            "longitude": -38.5 - 0.2 * (i % 7),
            "salinity_psu": float(round(rng.uniform(15.0, 42.0), 1)),
            "sra_run": f"SRR202{7265 + i}"
        })
    counts = rng.negative_binomial(n=5, p=0.3, size=(n_samples, n_asv))
    return {
        "records": records,
        "counts": counts,
        "features": [f"ASV_MANGROVE_{j:03d}" for j in range(n_asv)],
        "estuaries": estuaries
    }


@pytest.fixture
def temp_test_dir():
    """Provides a clean temporary directory for isolated file read/write tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir
