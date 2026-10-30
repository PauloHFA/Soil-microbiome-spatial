"""
src/data/mock_generator.py
Deterministic Synthetic Mock Dataset Generator for Soil Microbiome Pipeline.

Generates paired 16S and ITS profiles, ESDAC edaphic metadata, and FASTQ sequences
matching the empirical distribution of Labouyrie et al. (2023) Nature Communications.
"""

from __future__ import annotations

import argparse
import gzip
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class MockGeneratorConfig:
    """Configuration for synthetic mock dataset generation."""
    seed: int = 42
    n_samples: int = 715  # Default to 715 core Labouyrie cohort; or mini test cohort
    n_asvs_16s: int = 150
    n_asvs_its: int = 120
    output_dir: str = "data/mock"
    generate_fastqs: bool = False
    fastq_sample_limit: int = 4
    reads_per_fastq: int = 100


# Empirical distribution of 715 core samples from Labouyrie et al. (2023).
# land_cover_code is 0-indexed to match config/variable_schema.json (maximum: 5).
EMPIRICAL_CLASS_DISTRIBUTION: Dict[str, Dict[str, Any]] = {
    "Non-permanent crops": {"count": 264, "code": 0, "macro": "Cropland", "lucas_prefix": "B1"},
    "Permanent crops": {"count": 46, "code": 1, "macro": "Cropland", "lucas_prefix": "B7"},
    "Extensive grasslands": {"count": 128, "code": 2, "macro": "Grassland", "lucas_prefix": "E1"},
    "Intensive grassland": {"count": 18, "code": 3, "macro": "Grassland", "lucas_prefix": "E3"},
    "Coniferous forests": {"count": 160, "code": 4, "macro": "Woodland", "lucas_prefix": "C2"},
    "Broadleaved forests": {"count": 99, "code": 5, "macro": "Woodland", "lucas_prefix": "C1"},
}

# Re-sampled LUCAS sites. Both barcodes must share coordinates and site_id.
RESAMPLED_BARCODE_PAIRS: Dict[int, int] = {
    59: 138,
    102: 601,
    113: 717,
    116: 258,
}

# Molecular primers
PRIMERS: Dict[str, str] = {
    "16S_F": "GTGYCAGCMGCCGCGGTAA",       # 515F
    "16S_R": "GGCCGYCAATTYMTTTRAGTTT",    # 926R
    "ITS_F": "GTACACACCGCCCGTCG",          # ITS9mun
    "ITS_R": "CGCCTSCSCTTANTDATATGC",      # ITS4ngsUni
}


class MockDataGenerator:
    """Generates deterministic mock multi-marker datasets for testing and verification."""

    def __init__(self, config: Optional[MockGeneratorConfig] = None) -> None:
        self.config = config or MockGeneratorConfig()
        self.seed = self.config.seed
        self.rng = np.random.default_rng(self.seed)
        random.seed(self.seed)

    def _sample_class_counts(self, n_total: int) -> Dict[str, int]:
        """Compute class counts proportional to empirical distribution."""
        if n_total == 715:
            return {k: v["count"] for k, v in EMPIRICAL_CLASS_DISTRIBUTION.items()}

        total_empirical = 715
        counts: Dict[str, int] = {}
        assigned = 0
        classes = list(EMPIRICAL_CLASS_DISTRIBUTION.keys())
        for i, cls in enumerate(classes):
            if i == len(classes) - 1:
                counts[cls] = max(1, n_total - assigned)
            else:
                c = max(1, int(round(EMPIRICAL_CLASS_DISTRIBUTION[cls]["count"] / total_empirical * n_total)))
                counts[cls] = c
                assigned += c
        return counts

    def _cohort_barcodes(self, n_samples: int) -> List[int]:
        """
        Assign LUCAS barcodes for the synthetic cohort.

        Sequential 1..715 omits barcode 717, the re-sample of 113.
        The 715 core therefore keeps 1..714 and inserts 717.
        """
        if n_samples == 715:
            barcodes = list(range(1, 715)) + [717]
            missing = [barcode for barcode in RESAMPLED_BARCODE_PAIRS.values() if barcode not in barcodes]
            if missing:
                barcodes.extend(missing[:len(missing)])
            return barcodes[:n_samples]

        return list(range(1, n_samples + 1))

    def _colocate_resampled_sites(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bind each re-sampled pair to one site whenever both barcodes are present."""
        if df.empty or "barcode_id" not in df.columns:
            return df

        bound = df.copy()
        for initial_id, resampled_id in RESAMPLED_BARCODE_PAIRS.items():
            initial_mask = bound["barcode_id"] == initial_id
            resampled_mask = bound["barcode_id"] == resampled_id
            if not bool(initial_mask.any()) or not bool(resampled_mask.any()):
                continue

            shared = bound.loc[resampled_mask, ["latitude", "longitude"]].iloc[0]
            canonical_site = f"LUCAS_SITE_{initial_id}_{resampled_id}"
            pair_mask = initial_mask | resampled_mask
            bound.loc[pair_mask, "latitude"] = float(shared["latitude"])
            bound.loc[pair_mask, "longitude"] = float(shared["longitude"])
            bound.loc[pair_mask, "site_id"] = canonical_site
            if "spatial_group" in bound.columns:
                shared_group = bound.loc[resampled_mask, "spatial_group"].iloc[0]
                bound.loc[pair_mask, "spatial_group"] = shared_group
        return bound

    def generate_manifest(self) -> pd.DataFrame:
        """Generate paired sample manifest with realistic metadata and spatial coordinates."""
        class_counts = self._sample_class_counts(self.config.n_samples)
        records: List[Dict[str, Any]] = []
        barcodes = self._cohort_barcodes(sum(class_counts.values()))
        if len(barcodes) != sum(class_counts.values()):
            raise ValueError(
                f"Barcode list length {len(barcodes)} does not match cohort size {sum(class_counts.values())}"
            )
        barcode_iter = iter(barcodes)
        site_counter = 1

        for fine_class, count in class_counts.items():
            meta = EMPIRICAL_CLASS_DISTRIBUTION[fine_class]
            macro_class = meta["macro"]
            class_code = meta["code"]

            for _ in range(count):
                barcode_id = next(barcode_iter)
                sample_id = f"LUCAS_{barcode_id}"
                site_id = f"SITE_{site_counter:04d}"

                # Generate spatial coordinates across Europe
                if "crops" in fine_class.lower():
                    lat = float(self.rng.uniform(37.0, 56.0))
                    lon = float(self.rng.uniform(-6.0, 24.0))
                elif "grassland" in fine_class.lower():
                    lat = float(self.rng.uniform(42.0, 58.0))
                    lon = float(self.rng.uniform(-8.0, 22.0))
                elif "coniferous" in fine_class.lower():
                    lat = float(self.rng.uniform(52.0, 67.0))
                    lon = float(self.rng.uniform(5.0, 29.0))
                else:  # Broadleaved forests
                    lat = float(self.rng.uniform(44.0, 59.0))
                    lon = float(self.rng.uniform(-2.0, 25.0))

                # Physicochemical parameters conditioned on land cover
                if macro_class == "Cropland":
                    ph = float(np.clip(self.rng.normal(7.2, 0.5), 5.5, 8.8))
                    oc = float(np.clip(self.rng.normal(16.0, 5.0), 4.0, 45.0))
                    cn = float(np.clip(self.rng.normal(10.5, 1.2), 7.0, 16.0))
                    clay = float(np.clip(self.rng.normal(24.0, 6.0), 5.0, 55.0))
                    silt = float(np.clip(self.rng.normal(44.0, 8.0), 10.0, 70.0))
                    bulk_dens = float(np.clip(self.rng.normal(1.42, 0.10), 1.10, 1.75))
                elif macro_class == "Grassland":
                    ph = float(np.clip(self.rng.normal(6.3, 0.5), 4.8, 7.8))
                    oc = float(np.clip(self.rng.normal(42.0, 12.0), 15.0, 110.0))
                    cn = float(np.clip(self.rng.normal(12.8, 1.8), 8.5, 20.0))
                    clay = float(np.clip(self.rng.normal(22.0, 6.0), 5.0, 50.0))
                    silt = float(np.clip(self.rng.normal(38.0, 8.0), 10.0, 65.0))
                    bulk_dens = float(np.clip(self.rng.normal(1.24, 0.10), 0.95, 1.55))
                else:  # Woodland
                    ph = float(np.clip(self.rng.normal(4.7 if "coniferous" in fine_class.lower() else 5.8, 0.6), 3.4, 7.5))
                    oc = float(np.clip(self.rng.normal(55.0 if "coniferous" in fine_class.lower() else 38.0, 15.0), 18.0, 180.0))
                    cn = float(np.clip(self.rng.normal(21.0 if "coniferous" in fine_class.lower() else 15.0, 3.0), 11.0, 35.0))
                    clay = float(np.clip(self.rng.normal(17.0, 5.0), 3.0, 45.0))
                    silt = float(np.clip(self.rng.normal(34.0, 7.0), 8.0, 60.0))
                    bulk_dens = float(np.clip(self.rng.normal(1.08, 0.12), 0.70, 1.45))

                sand = float(max(2.0, 100.0 - (clay + silt)))
                text_sum = clay + silt + sand
                clay = round(clay / text_sum * 100.0, 2)
                silt = round(silt / text_sum * 100.0, 2)
                sand = round(100.0 - clay - silt, 2)

                nitrogen = round(oc / cn, 3)
                ph_cacl2 = round(ph - float(self.rng.uniform(0.55, 0.85)), 2)

                run_16s = f"SRR2406{barcode_id:04d}"
                run_its = f"SRR2407{barcode_id:04d}"
                reads_16s = int(self.rng.lognormal(mean=11.6, sigma=0.4))
                reads_its = int(self.rng.lognormal(mean=8.9, sigma=0.5))

                spatial_group = int(site_counter % 5) + 1

                records.append({
                    "sample_id": sample_id,
                    "barcode_id": barcode_id,
                    "site_id": site_id,
                    "point_id": 100000 + barcode_id,
                    "latitude": round(lat, 5),
                    "longitude": round(lon, 5),
                    "spatial_group": spatial_group,
                    "land_cover": macro_class,
                    "macro_land_cover": macro_class,
                    "fine_land_cover": fine_class,
                    "land_cover_code": class_code,
                    "sra_16s_run": run_16s,
                    "reads_16s": reads_16s,
                    "sra_its_run": run_its,
                    "reads_its": reads_its,
                    "esdac_ph": round(ph, 2),
                    "ph_in_cacl2": ph_cacl2,
                    "esdac_oc": round(oc, 2),
                    "esdac_n": nitrogen,
                    "c_n_ratio": round(cn, 2),
                    "esdac_clay": clay,
                    "esdac_silt": silt,
                    "esdac_sand": sand,
                    "bulk_density": round(bulk_dens, 2),
                    "is_paired": True,
                    "qc_status": "PASS",
                    "include_in_core": True,
                    "study_accession": "PRJNA952168",
                })

                site_counter += 1

        return self._colocate_resampled_sites(pd.DataFrame(records))

    def generate_asv_matrices(
        self, manifest: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate ASV abundance count matrices and taxonomy annotations.
        Conditioned on land cover and soil pH for realistic biological signals.
        """
        sample_ids = manifest["sample_id"].tolist()
        n_samples = len(sample_ids)
        macro_labels = manifest["macro_land_cover"].values
        phs = manifest["esdac_ph"].values

        # 1. Bacterial 16S ASV matrix
        n_16s = self.config.n_asvs_16s
        asv_16s_cols = [f"ASV_16S_{i:04d}" for i in range(1, n_16s + 1)]
        counts_16s = np.zeros((n_samples, n_16s), dtype=int)

        tax_16s = []
        for i, asv in enumerate(asv_16s_cols):
            if i < n_16s * 0.30:
                phylum, tax = "p__Acidobacteriota", f"d__Bacteria;p__Acidobacteriota;c__Acidobacteriae;o__Acidobacteriales;f__Acidobacteriaceae;g__Bryocella;s__{asv}"
            elif i < n_16s * 0.60:
                phylum, tax = "p__Actinomycetota", f"d__Bacteria;p__Actinomycetota;c__Actinomycetes;o__Micrococcales;f__Micrococcaceae;g__Arthrobacter;s__{asv}"
            elif i < n_16s * 0.85:
                phylum, tax = "p__Pseudomonadota", f"d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;o__Pseudomonadales;f__Pseudomonadaceae;g__Pseudomonas;s__{asv}"
            else:
                phylum, tax = "p__Bacteroidota", f"d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Flavobacteriales;f__Flavobacteriaceae;g__Flavobacterium;s__{asv}"
            tax_16s.append({"asv_id": asv, "marker": "16S", "phylum": phylum, "taxonomy": tax})

        for row_idx in range(n_samples):
            ph = phs[row_idx]
            depth = manifest["reads_16s"].iloc[row_idx]
            probs = np.zeros(n_16s)
            probs[0:int(n_16s * 0.30)] = np.exp(-0.7 * (ph - 4.5))
            probs[int(n_16s * 0.30):int(n_16s * 0.60)] = np.exp(0.5 * (ph - 6.5))
            probs[int(n_16s * 0.60):] = self.rng.gamma(shape=1.5, scale=1.0, size=n_16s - int(n_16s * 0.60))
            # Induce realistic ecological sparsity (taxa absent in this sample)
            absent_mask = self.rng.random(n_16s) < 0.50
            probs[absent_mask] = 0.0
            probs_sum = probs.sum()
            probs = probs / probs_sum if probs_sum > 0 else np.ones(n_16s) / n_16s
            counts_16s[row_idx, :] = self.rng.multinomial(n=min(depth, 50000), pvals=probs)

        df_counts_16s = pd.DataFrame(counts_16s, index=sample_ids, columns=asv_16s_cols)
        df_tax_16s = pd.DataFrame(tax_16s)

        # 2. Fungal ITS ASV matrix
        n_its = self.config.n_asvs_its
        asv_its_cols = [f"ASV_ITS_{i:04d}" for i in range(1, n_its + 1)]
        counts_its = np.zeros((n_samples, n_its), dtype=int)

        tax_its = []
        for i, asv in enumerate(asv_its_cols):
            if i < n_its * 0.45:
                phylum, tax = "p__Basidiomycota", f"k__Fungi;p__Basidiomycota;c__Agaricomycetes;o__Agaricales;f__Cortinariaceae;g__Cortinarius;s__{asv}"
            elif i < n_its * 0.85:
                phylum, tax = "p__Ascomycota", f"k__Fungi;p__Ascomycota;c__Sordariomycetes;o__Hypocreales;f__Nectriaceae;g__Fusarium;s__{asv}"
            else:
                phylum, tax = "p__Mortierellomycota", f"k__Fungi;p__Mortierellomycota;c__Mortierellomycetes;o__Mortierellales;f__Mortierellaceae;g__Mortierella;s__{asv}"
            tax_its.append({"asv_id": asv, "marker": "ITS", "phylum": phylum, "taxonomy": tax})

        for row_idx in range(n_samples):
            macro = macro_labels[row_idx]
            depth = manifest["reads_its"].iloc[row_idx]
            probs = np.zeros(n_its)
            if macro == "Woodland":
                probs[0:int(n_its * 0.45)] = self.rng.gamma(shape=3.0, scale=1.5, size=int(n_its * 0.45))
                probs[int(n_its * 0.45):] = self.rng.gamma(shape=0.8, scale=1.0, size=n_its - int(n_its * 0.45))
            elif macro == "Cropland":
                probs[0:int(n_its * 0.45)] = self.rng.gamma(shape=0.5, scale=1.0, size=int(n_its * 0.45))
                probs[int(n_its * 0.45):] = self.rng.gamma(shape=2.5, scale=1.5, size=n_its - int(n_its * 0.45))
            else:
                probs = self.rng.gamma(shape=1.5, scale=1.0, size=n_its)
            # Induce realistic ecological sparsity (taxa absent in this sample)
            absent_mask_its = self.rng.random(n_its) < 0.50
            probs[absent_mask_its] = 0.0
            probs_sum = probs.sum()
            probs = probs / probs_sum if probs_sum > 0 else np.ones(n_its) / n_its
            counts_its[row_idx, :] = self.rng.multinomial(n=min(depth, 3000), pvals=probs)

        df_counts_its = pd.DataFrame(counts_its, index=sample_ids, columns=asv_its_cols)
        df_tax_its = pd.DataFrame(tax_its)

        return df_counts_16s, df_counts_its, df_tax_16s, df_tax_its

    def generate_mock_fastqs(self, manifest: pd.DataFrame, out_dir: Path) -> List[Tuple[str, str]]:
        """
        Generate paired-end 16S FASTQ files with TruSeq dual-orientation amplicons
        and single-end ITS FASTQ files for a subset of samples.
        """
        fastq_dir = out_dir / "raw_fastqs"
        fastq_dir.mkdir(parents=True, exist_ok=True)
        generated_files: List[Tuple[str, str]] = []

        sub_samples = manifest.head(self.config.fastq_sample_limit)

        for _, row in sub_samples.iterrows():
            sample_id = row["sample_id"]
            run_16s = row["sra_16s_run"]
            run_its = row["sra_its_run"]

            r1_path = fastq_dir / f"{run_16s}_1.fastq.gz"
            r2_path = fastq_dir / f"{run_16s}_2.fastq.gz"

            with gzip.open(r1_path, "wt") as f1, gzip.open(r2_path, "wt") as f2:
                for r in range(self.config.reads_per_fastq):
                    read_id = f"@{run_16s}.{r + 1} {sample_id} length=300"
                    is_forward = (r % 2 == 0)
                    insert_len = 260
                    rand_bases = "".join(self.rng.choice(["A", "C", "G", "T"], size=insert_len))

                    if is_forward:
                        seq_r1 = PRIMERS["16S_F"] + rand_bases[:280]
                        seq_r2 = PRIMERS["16S_R"] + rand_bases[-280:]
                    else:
                        seq_r1 = PRIMERS["16S_R"] + rand_bases[:280]
                        seq_r2 = PRIMERS["16S_F"] + rand_bases[-280:]

                    q_score = "I" * len(seq_r1)

                    f1.write(f"{read_id}\n{seq_r1}\n+\n{q_score}\n")
                    f2.write(f"{read_id}\n{seq_r2}\n+\n{q_score}\n")

            its_path = fastq_dir / f"{run_its}.fastq.gz"
            with gzip.open(its_path, "wt") as f_its:
                for r in range(self.config.reads_per_fastq):
                    read_id = f"@{run_its}.{r + 1} {sample_id}_ITS length=600"
                    insert_len = 550
                    rand_bases = "".join(self.rng.choice(["A", "C", "G", "T"], size=insert_len))
                    seq_its = PRIMERS["ITS_F"] + rand_bases + PRIMERS["ITS_R"]
                    q_score = "J" * len(seq_its)
                    f_its.write(f"{read_id}\n{seq_its}\n+\n{q_score}\n")

            generated_files.append((str(r1_path), str(its_path)))

        return generated_files

    def generate_all(self) -> Dict[str, str]:
        """Execute full synthetic generation and save artifacts."""
        out_path = Path(self.config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        manifest = self.generate_manifest()
        df_16s, df_its, tax_16s, tax_its = self.generate_asv_matrices(manifest)

        manifest_file = out_path / "paired_sample_manifest.csv"
        manifest.to_csv(manifest_file, index=False)

        counts_16s_file = out_path / "bacteria_16s_asv_matrix.csv"
        df_16s.to_csv(counts_16s_file)

        counts_its_file = out_path / "fungi_its_asv_matrix.csv"
        df_its.to_csv(counts_its_file)

        tax_16s_file = out_path / "taxonomy_16s.csv"
        tax_16s.to_csv(tax_16s_file, index=False)

        tax_its_file = out_path / "taxonomy_its.csv"
        tax_its.to_csv(tax_its_file, index=False)

        result_paths = {
            "manifest": str(manifest_file),
            "counts_16s": str(counts_16s_file),
            "counts_its": str(counts_its_file),
            "taxonomy_16s": str(tax_16s_file),
            "taxonomy_its": str(tax_its_file),
        }

        if self.config.generate_fastqs:
            self.generate_mock_fastqs(manifest, out_path)
            result_paths["fastqs_dir"] = str(out_path / "raw_fastqs")

        return result_paths


def generate_mock_data(
    n_samples: int = 715,
    seed: int = 42,
    output_dir: str = "data/mock",
    generate_fastqs: bool = False
) -> Dict[str, str]:
    """Functional convenience entrypoint for mock generation."""
    cfg = MockGeneratorConfig(
        n_samples=n_samples,
        seed=seed,
        output_dir=output_dir,
        generate_fastqs=generate_fastqs
    )
    generator = MockDataGenerator(cfg)
    return generator.generate_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic Soil Microbiome Mock Data Generator")
    parser.add_argument("--cohort", type=int, default=715, help="Sample cohort size (715 or mini e.g. 12)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--out-dir", type=str, default="data/mock", help="Output directory path")
    parser.add_argument("--fastqs", action="store_true", help="Generate synthetic FASTQs")
    args = parser.parse_args()

    paths = generate_mock_data(
        n_samples=args.cohort,
        seed=args.seed,
        output_dir=args.out_dir,
        generate_fastqs=args.fastqs
    )
    print("Mock dataset generation completed successfully:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
