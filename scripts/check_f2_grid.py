import pandas as pd

f7 = "data/downloads/41467_2023_37937_MOESM7_ESM.xlsx"
df_f2 = pd.read_excel(f7, sheet_name="Figure2")
print("Row 0, cols 4 to 12:")
print(df_f2.iloc[0, 4:12].tolist())
print("Row 1, cols 4 to 12:")
print(df_f2.iloc[1, 4:12].tolist())
print("Row 2, cols 4 to 12:")
print(df_f2.iloc[2, 4:12].tolist())

