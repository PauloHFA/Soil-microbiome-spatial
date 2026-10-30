import pandas as pd

f5 = "data/downloads/41467_2023_37937_MOESM5_ESM.xlsx"
print("=== MOESM5: Observed_richness ===")
df_obs = pd.read_excel(f5, sheet_name="Observed_richness")
print(df_obs.info())
print(df_obs.head())

print("\n=== MOESM5: Shannon_index ===")
df_shan = pd.read_excel(f5, sheet_name="Shannon_index")
print(df_shan.info())
print(df_shan.head())

f7 = "data/downloads/41467_2023_37937_MOESM7_ESM.xlsx"
print("\n=== MOESM7: Figure2 ===")
df_f2 = pd.read_excel(f7, sheet_name="Figure2")
print(df_f2.info())
print(df_f2.head())

