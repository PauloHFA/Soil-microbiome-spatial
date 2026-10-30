import pandas as pd
import numpy as np
from pathlib import Path

out_dir = Path("data/real_cohort_715")
out_dir.mkdir(parents=True, exist_ok=True)

f7 = "data/downloads/41467_2023_37937_MOESM7_ESM.xlsx"
df_f2 = pd.read_excel(f7, sheet_name="Figure2")

# Block 1: Alpha diversity & Land cover (rows 2 to 716 -> exactly 715 samples)
print("Extracting Block 1 (Alpha diversity & Land cover)...")
b1 = df_f2.iloc[2:717, :5].copy()
b1.columns = ["richness_16s", "shannon_16s", "richness_its", "shannon_its", "Sub_level"]
# Note that row 2:717 corresponds to the 715 samples in order
sample_names = df_f2.iloc[721, 1:716].tolist()
print(f"Sample names in order: {len(sample_names)}, first 5: {sample_names[:5]}, last 5: {sample_names[-5:]}")
b1.index = sample_names
b1.to_csv(out_dir / "real_alpha_diversity_715.csv")

# Block 2: Bacterial Bray-Curtis / Distance Matrix (rows 722 to 1436)
print("Extracting Block 2 (Bacterial Distance Matrix 715x715)...")
dist_bact = df_f2.iloc[722:1437, 1:716].values.astype(float)
df_dist_bact = pd.DataFrame(dist_bact, index=sample_names, columns=sample_names)
df_dist_bact.to_csv(out_dir / "real_distance_matrix_16s_715.csv")
print(f"Bacterial distance matrix shape: {df_dist_bact.shape}")

# Block 3: Fungal Bray-Curtis / Distance Matrix (rows 1442 to 2156)
print("Extracting Block 3 (Fungal Distance Matrix 715x715)...")
dist_fungi = df_f2.iloc[1442:2157, 1:716].values.astype(float)
df_dist_fungi = pd.DataFrame(dist_fungi, index=sample_names, columns=sample_names)
df_dist_fungi.to_csv(out_dir / "real_distance_matrix_its_715.csv")
print(f"Fungal distance matrix shape: {df_dist_fungi.shape}")

print("\nSUCCESS! Extracted all 715 real paired distance matrices & alpha diversity tables.")
