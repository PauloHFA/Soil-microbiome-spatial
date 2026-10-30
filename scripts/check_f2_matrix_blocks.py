import pandas as pd

f7 = "data/downloads/41467_2023_37937_MOESM7_ESM.xlsx"
df_f2 = pd.read_excel(f7, sheet_name="Figure2")

print("Row 718:", df_f2.iloc[718, :5].tolist())
print("Row 719:", df_f2.iloc[719, :5].tolist())
print("Row 1430:", df_f2.iloc[1430, :5].tolist())
print("Row 1431:", df_f2.iloc[1431, :5].tolist())
print("Row 1434:", df_f2.iloc[1434, :5].tolist())

