import pandas as pd

f7 = "data/downloads/41467_2023_37937_MOESM7_ESM.xlsx"
xl = pd.ExcelFile(f7)

# Check Suppfigure1, Suppfigure2, Suppfigure3
for s in ["Suppfigure1", "Suppfigure2", "Suppfigure3"]:
    df = xl.parse(s)
    print(f"\n=== Sheet: {s} ===")
    print("Shape:", df.shape)
    print(df.head(5))

