import pandas as pd

f7 = "data/downloads/41467_2023_37937_MOESM7_ESM.xlsx"
df_f2 = pd.read_excel(f7, sheet_name="Figure2")

print("Row 720 header:", df_f2.iloc[720, :6].tolist())
print("Row 721 first row of block 2:", df_f2.iloc[721, :6].tolist())
print("Row 1433:", df_f2.iloc[1433, :6].tolist())
print("Row 1434 header for block 3?:", df_f2.iloc[1434, :6].tolist())
print("Row 1435:", df_f2.iloc[1435, :6].tolist())

