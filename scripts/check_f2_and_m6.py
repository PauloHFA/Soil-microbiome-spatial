import pandas as pd
import numpy as np

f7 = "data/downloads/41467_2023_37937_MOESM7_ESM.xlsx"
df_f2 = pd.read_excel(f7, sheet_name="Figure2")
print("Figure2 shape:", df_f2.shape)
print("Figure2 first rows:")
print(df_f2.iloc[:10, :5])

f6 = "data/downloads/41467_2023_37937_MOESM6_ESM.xlsx"
xl6 = pd.ExcelFile(f6)
for s in xl6.sheet_names:
    print(f"\nSheet in MOESM6: {s}")
    d = xl6.parse(s, nrows=5)
    print(d.head())

