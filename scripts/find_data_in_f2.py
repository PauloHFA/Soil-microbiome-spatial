import pandas as pd

f7 = "data/downloads/41467_2023_37937_MOESM7_ESM.xlsx"
df_f2 = pd.read_excel(f7, sheet_name="Figure2")
# Find where the data lives in cols 5 to 715
col_samples = []
for c in range(df_f2.shape[1]):
    val = df_f2.iloc[:, c].dropna()
    if len(val) > 0:
        first_few = val.head(3).tolist()
        last_few = val.tail(3).tolist()
        if c in [0, 1, 2, 3, 4, 5, 6, 10, 100, 715]:
            print(f"Col {c}: len={len(val)}, first={first_few}, last={last_few}")

