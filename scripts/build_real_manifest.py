import pandas as pd
import requests, io
from pathlib import Path

out_dir = Path("data/real_cohort_715")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Load Nature Suppfigure16
f7 = "data/downloads/41467_2023_37937_MOESM7_ESM.xlsx"
df16 = pd.read_excel(f7, sheet_name="Suppfigure16")
header = [str(c) if pd.notna(c) else "sample_code" for c in df16.iloc[1].tolist()]
df = df16.iloc[2:].copy()
df.columns = header
df["barcode_id"] = df["sample_code"].str.extract(r"Sample(\d+)")[0].astype(int)

# Mapping Sub_level to clean English category names matching project specification
sub_map = {
    "1.Coniferous wood": "Coniferous forests",
    "2.Broadleaved wood": "Broadleaved forests",
    "3.Extensive grassland": "Extensive grasslands",
    "4.Intensive grassland": "Intensive grassland",
    "5.Permanent crop": "Permanent crops",
    "6.Non permanent crop": "Non-permanent crops"
}
df["fine_land_cover"] = df["Sub_level"].map(sub_map)

# 2. Fetch ENA real run accessions & coordinates
print("Fetching ENA coordinates & runs for PRJNA952168...")
url = "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJNA952168&result=read_run&fields=run_accession,sample_accession,sample_title,country,location,instrument_model,library_layout,read_count&format=tsv"
r = requests.get(url, timeout=45)
df_ena = pd.read_csv(io.StringIO(r.text), sep="\t")
df_ena["barcode_id"] = df_ena["sample_title"].str.extract(r"LUCAS_(\d+)")[0].dropna().astype(int)

# 16S and ITS separation
df_16s = df_ena[df_ena["library_layout"] == "PAIRED"].sort_values("read_count", ascending=False).drop_duplicates("barcode_id")
df_its = df_ena[df_ena["library_layout"] == "SINGLE"].sort_values("read_count", ascending=False).drop_duplicates("barcode_id")

# Merge
m = df.merge(df_16s[["barcode_id", "run_accession", "read_count", "country", "location"]].rename(
    columns={"run_accession": "run_16s", "read_count": "reads_16s"}
), on="barcode_id")
m = m.merge(df_its[["barcode_id", "run_accession", "read_count"]].rename(
    columns={"run_accession": "run_its", "read_count": "reads_its"}
), on="barcode_id")

m["sample_id"] = "LUCAS_" + m["barcode_id"].astype(str)
m["site_id"] = "SITE_" + m["barcode_id"].astype(str)

# Parse lat/lon from location if available, otherwise fallback
def parse_loc(loc_str):
    if pd.isna(loc_str) or not str(loc_str).strip():
        return None, None
    parts = str(loc_str).split()
    # e.g. 41.46242 N 5.31853 W
    try:
        lat = float(parts[0]) * (-1 if len(parts) > 1 and parts[1].upper() == "S" else 1)
        lon = float(parts[2]) * (-1 if len(parts) > 3 and parts[3].upper() == "W" else 1)
        return lat, lon
    except Exception:
        return None, None

coords = [parse_loc(l) for l in m["location"]]
m["latitude"] = [c[0] for c in coords]
m["longitude"] = [c[1] for c in coords]

# Save manifest
manifest_path = out_dir / "real_paired_manifest_715.csv"
m.to_csv(manifest_path, index=False)
print(f"Saved real cohort manifest: {manifest_path} ({len(m)} samples)")
print(m[["barcode_id", "fine_land_cover", "pH_H2O", "temp_seasonality", "reads_16s", "reads_its", "country"]].head())

