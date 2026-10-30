import requests
from pathlib import Path
import openpyxl

out_dir = Path("data/downloads")
out_dir.mkdir(parents=True, exist_ok=True)

files = [
    "41467_2023_37937_MOESM5_ESM.xlsx",
    "41467_2023_37937_MOESM6_ESM.xlsx",
    "41467_2023_37937_MOESM7_ESM.xlsx",
]

for fname in files:
    url = f"https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-023-37937-4/MediaObjects/{fname}"
    dest = out_dir / fname
    if not dest.exists():
        print(f"Downloading {fname}...")
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        dest.write_bytes(r.content)
        print(f"Saved {fname} ({len(r.content)} bytes)")
    
    wb = openpyxl.load_workbook(dest, read_only=True)
    print(f"\n=== {fname} sheets ===")
    print(wb.sheetnames)

