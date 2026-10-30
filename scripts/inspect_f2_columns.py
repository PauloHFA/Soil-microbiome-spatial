import pandas as pd

f7 = "data/downloads/41467_2023_37937_MOESM7_ESM.xlsx"
df_f2 = pd.read_excel(f7, sheet_name="Figure2")
# Find where the 716 columns come from
print("Columns with non-null values in row 1:")
for col_idx in range(df_f2.shape[1]):
    val = df_f2.iloc[1, col_idx]
    if pd.notna(val):
        print(f"Col {col_idx}: {val}")

# Check non-null values across columns
cols_with_data = [c for c in range(df_f2.shape[1]) if df_f2.iloc[:, c].notna().sum() > 10]
print(f"Total columns with >10 non-null rows: {len(cols_with_data)}")
for c in cols_with_data[:15]:
    print(f"Col {c} header: {df_f2.iloc[1, c]} (rows non-null: {df_f2.iloc[:, c].notna().sum()})")

