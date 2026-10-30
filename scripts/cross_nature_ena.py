import pandas as pd
import requests, io

# 1. Load Nature Suppfigure16 (The 715 cohort metadata)
f7 = "data/downloads/41467_2023_37937_MOESM7_ESM.xlsx"
df16 = pd.read_excel(f7, sheet_name="Suppfigure16")
header = [str(c) if pd.notna(c) else "sample_code" for c in df16.iloc[1].tolist()]
df16_clean = df16.iloc[2:].copy()
df16_clean.columns = header
df16_clean["barcode_id"] = df16_clean["sample_code"].str.extract(r"Sample(\d+)")[0].astype(int)
print(f"Cohort 715 samples from Nature: {len(df16_clean)}")
print(df16_clean[["barcode_id", "Sub_level", "pH_H2O", "temp_seasonality", "Observed_richness", "Shannon_index"]].head())

# 2. Fetch ENA real run accessions
url = "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJNA952168&result=read_run&fields=run_accession,sample_accession,sample_title,country,location,instrument_model,library_layout,read_count&format=tsv"
r = requests.get(url, timeout=45)
df_ena = pd.read_csv(io.StringIO(r.text), sep="\t")
df_ena["barcode_id"] = df_ena["sample_title"].str.extract(r"LUCAS_(\d+)")[0].dropna().astype(int)

# Separate 16S (Illumina MiSeq, PAIRED) and ITS (Sequel II, SINGLE)
df_16s = df_ena[df_ena["library_layout"] == "PAIRED"].sort_values("read_count", ascending=False).drop_duplicates("barcode_id")
df_its = df_ena[df_ena["library_layout"] == "SINGLE"].sort_values("read_count", ascending=False).drop_duplicates("barcode_id")

# Merge with the 715 Nature cohort
merged = df16_clean.merge(df_16s[["barcode_id", "run_accession", "read_count"]].rename(columns={"run_accession": "run_16s", "read_count": "reads_16s"}), on="barcode_id")
merged = merged.merge(df_its[["barcode_id", "run_accession", "read_count"]].rename(columns={"run_accession": "run_its", "read_count": "reads_its"}), on="barcode_id")

print(f"\nSuccessfully matched cohort samples with ENA SRA runs: {len(merged)} / {len(df16_clean)}")
print(merged[["barcode_id", "Sub_level", "pH_H2O", "run_16s", "reads_16s", "run_its", "reads_its"]].head(10))

