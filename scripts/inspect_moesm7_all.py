import pandas as pd

f7 = "data/downloads/41467_2023_37937_MOESM7_ESM.xlsx"
xl = pd.ExcelFile(f7)
for sheet in xl.sheet_names:
    df_head = xl.parse(sheet, nrows=5)
    print(f"\n--- Sheet: {sheet} (shape: {df_head.shape}) ---")
    print(df_head.iloc[:3, :6])

