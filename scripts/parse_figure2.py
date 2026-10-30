import pandas as pd

f7 = "data/downloads/41467_2023_37937_MOESM7_ESM.xlsx"
df_f2 = pd.read_excel(f7, sheet_name="Figure2")
print("Figure 2 header row:")
print(df_f2.iloc[1, :10].tolist())
print("\nUnique metric names in column 0:")
print(df_f2.iloc[1:, 0].value_counts())
print("\nRows sample:")
print(df_f2.iloc[1:6, :8])

