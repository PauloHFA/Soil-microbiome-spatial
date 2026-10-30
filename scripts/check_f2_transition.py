import pandas as pd

f7 = "data/downloads/41467_2023_37937_MOESM7_ESM.xlsx"
df_f2 = pd.read_excel(f7, sheet_name="Figure2")

for r in range(1430, 1445):
    val0 = df_f2.iloc[r, 0]
    val1 = df_f2.iloc[r, 1]
    if pd.isna(val0) or "Figure" in str(val0) or "Sample" in str(val0):
        print(f"Row {r}: col0={val0}, col1={val1}")

