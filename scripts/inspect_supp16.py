import pandas as pd

f7 = "data/downloads/41467_2023_37937_MOESM7_ESM.xlsx"
df16 = pd.read_excel(f7, sheet_name="Suppfigure16")
print("=== Suppfigure16 info ===")
print(df16.info())
print("\nFirst 10 rows:")
print(df16.head(10))

