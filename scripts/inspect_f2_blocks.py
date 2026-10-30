import pandas as pd

f7 = "data/downloads/41467_2023_37937_MOESM7_ESM.xlsx"
df_f2 = pd.read_excel(f7, sheet_name="Figure2")
print("Total rows in Figure2:", len(df_f2))
print("Row 0: 5 values:", df_f2.iloc[0, :5].tolist())
print("Row 1: 5 values:", df_f2.iloc[1, :5].tolist())
print("Row 2: 5 values:", df_f2.iloc[2, :5].tolist())
print("Row 715: 5 values:", df_f2.iloc[715, :5].tolist())
print("Row 716: 5 values:", df_f2.iloc[716, :5].tolist())
print("Row 717: 5 values:", df_f2.iloc[717, :5].tolist())
print("Row 1432: 5 values:", df_f2.iloc[1432, :5].tolist())
print("Row 1433: 5 values:", df_f2.iloc[1433, :5].tolist())

