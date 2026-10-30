"""
incorporar_dados_esdac_real.py - Integra metadados oficiais do ESDAC (físico-química de solo e clima)
à coorte real de 715 sítios europeus e executa a discriminação espacial multimodal.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.data.esdac_parser import ESDACLucasParser
from src.spatial_cv.spatial_partitioner import create_spatial_splits
from src.spatial_cv.leakage_guard import assert_zero_spatial_leakage
from src.evaluation.bootstrap_ci import compute_spatial_bootstrap_ci
from src.evaluation.hypothesis_test import evaluate_multimarker_hypothesis
from src.ecology.multivariate_stats import run_permanova

print("--- 1. Carregando dados microbiológicos e planilha ESDAC ---")
data_dir = ROOT / "data" / "real_cohort_715"
out_dir = ROOT / "results" / "real_analysis_715"
out_dir.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(data_dir / "real_paired_manifest_715.csv")
d_16s = pd.read_csv(data_dir / "real_distance_matrix_16s_715.csv", index_col=0).values
d_its = pd.read_csv(data_dir / "real_distance_matrix_its_715.csv", index_col=0).values
alpha = pd.read_csv(data_dir / "real_alpha_diversity_715.csv", index_col=0)

excel_path = ROOT / "data" / "METADATA_LUCAS18_BIODIVERSITY_885ID_COMPLETED.xlsx"
parser = ESDACLucasParser(excel_path)
df_esdac = parser.parse()

# Selecionar variáveis edáficas do ESDAC
edaphic_cols = [
    "barcode_id", "ph_h2o", "ph_cacl2", "organic_carbon", "nitrogen",
    "c_n_ratio", "phosphorus", "potassium", "caco3", "clay", "silt", "sand",
    "bulk_density", "coarse_fragments", "ec"
]
df_soil_sub = df_esdac[edaphic_cols].copy()

# Fazer merge mantendo os 715 sítios pareados
manifest_with_soil = pd.merge(manifest, df_soil_sub, on="barcode_id", how="left")

# Salvar manifesto atualizado enriquecido com solo
manifest_with_soil.to_csv(data_dir / "real_paired_manifest_715_with_soil.csv", index=False)
print(f"Manifesto com solo salvo: {len(manifest_with_soil)} amostras.")

# 2. Embeddings PCoA
print("\n--- 2. Calculando Embeddings PCoA (16S e ITS) ---")
mds_16s = MDS(n_components=30, dissimilarity="precomputed", random_state=42)
X_16s = mds_16s.fit_transform(d_16s)

mds_its = MDS(n_components=30, dissimilarity="precomputed", random_state=42)
X_its = mds_its.fit_transform(d_its)

X_integrated_markers = np.hstack([X_16s, X_its])

# Preparar matriz edáfica X_soil
soil_feature_names = [
    "ph_h2o", "ph_cacl2", "organic_carbon", "nitrogen", "c_n_ratio",
    "phosphorus", "potassium", "caco3", "clay", "silt", "sand",
    "bulk_density", "coarse_fragments", "ec"
]
X_soil_raw = manifest_with_soil[soil_feature_names].values

# Classes
classes = sorted(manifest["fine_land_cover"].unique().tolist())
class_to_code = {c: i for i, c in enumerate(classes)}
y = manifest["fine_land_cover"].map(class_to_code).values

print(f"Distribuição de classes (6 fitofisionomias):")
for c, cnt in manifest["fine_land_cover"].value_counts().items():
    print(f"  {c:25}: {cnt}")

# 3. Particionamento espacial estrito (5 Folds sem vazamento por país)
splits = create_spatial_splits(manifest, n_splits=5, seed=42, group_col="country")
for tr, te in splits:
    tr_c = set(manifest.iloc[tr]["country"].dropna())
    te_c = set(manifest.iloc[te]["country"].dropna())
    assert_zero_spatial_leakage(tr_c, te_c)

print("\nZero vazamento espacial estrito verificado nas 5 dobras!")

# 4. Modelagem com 5 conjuntos de características:
# Set A: 16S Bactérias
# Set B: ITS Fungos
# Set C: Multi-marcador (16S + ITS)
# Set D: Físico-química de Solo Pura (Edaphic)
# Set E: Holístico Multimodal (16S + ITS + Solo)

feature_sets_names = [
    "set_a_16s",
    "set_b_its",
    "set_c_integrated",
    "set_d_soil",
    "set_e_holistic"
]

fold_scores = {s: [] for s in feature_sets_names}
train_scores = {s: [] for s in feature_sets_names}
all_preds = {s: [] for s in feature_sets_names}
all_preds["y_true"] = []

for fold_idx, (tr, te) in enumerate(splits):
    y_tr, y_te = y[tr], y[te]
    all_preds["y_true"].extend(y_te.tolist())
    
    # Imputar e padronizar variáveis de solo usando APENAS o treino da dobra (Zero Leakage)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    X_soil_tr = scaler.fit_transform(imputer.fit_transform(X_soil_raw[tr]))
    X_soil_te = scaler.transform(imputer.transform(X_soil_raw[te]))
    
    # Montar matrizes para a dobra
    fold_features = {
        "set_a_16s": (X_16s[tr], X_16s[te]),
        "set_b_its": (X_its[tr], X_its[te]),
        "set_c_integrated": (X_integrated_markers[tr], X_integrated_markers[te]),
        "set_d_soil": (X_soil_tr, X_soil_te),
        "set_e_holistic": (
            np.hstack([X_integrated_markers[tr], X_soil_tr]),
            np.hstack([X_integrated_markers[te], X_soil_te])
        )
    }
    
    for s_name in feature_sets_names:
        X_tr, X_te = fold_features[s_name]
        clf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42 + fold_idx, n_jobs=-1
        )
        clf.fit(X_tr, y_tr)
        
        pred_tr = clf.predict(X_tr)
        pred_te = clf.predict(X_te)
        
        tr_acc = balanced_accuracy_score(y_tr, pred_tr)
        te_acc = balanced_accuracy_score(y_te, pred_te)
        
        train_scores[s_name].append(float(tr_acc))
        fold_scores[s_name].append(float(te_acc))
        all_preds[s_name].extend(pred_te.tolist())

# 5. Métricas Consolidadas e Bootstrap CIs
print("\n--- 5. Avaliação Estatística e Testes de Hipótese ---")
results = {}
for s_name in feature_sets_names:
    y_true_arr = np.array(all_preds["y_true"])
    y_pred_arr = np.array(all_preds[s_name])
    
    ci_res = compute_spatial_bootstrap_ci(np.array(fold_scores[s_name]), n_bootstraps=1000, seed=42)
    ci_low = ci_res["ci_lower"]
    ci_high = ci_res["ci_upper"]
    mean_fold_acc = float(np.mean(fold_scores[s_name]))
    std_fold_acc = float(np.std(fold_scores[s_name]))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro"))
    
    results[s_name] = {
        "mean_balanced_accuracy": mean_fold_acc,
        "std_balanced_accuracy": std_fold_acc,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "macro_f1": macro_f1,
        "fold_scores": fold_scores[s_name],
        "train_scores": train_scores[s_name]
    }
    print(f"[{s_name:18}] Acurácia Balanceada: {mean_fold_acc:.4f} (IC 95%: {ci_low:.4f} - {ci_high:.4f}) | Macro-F1: {macro_f1:.4f}")

# Teste de hipótese H0: Modelo Holístico vs Melhor Modelo Individual
hyp_test = evaluate_multimarker_hypothesis(
    scores_16s=np.array(fold_scores["set_a_16s"]),
    scores_its=np.array(fold_scores["set_b_its"]),
    scores_integrated=np.array(fold_scores["set_e_holistic"]),
    training_scores_16s=train_scores["set_a_16s"],
    training_scores_its=train_scores["set_b_its"],
)
results["hypothesis_test_holistic"] = hyp_test
print(f"\nTeste de Hipótese (Holístico vs Melhor Marcador):")
print(f"  Delta: {hyp_test['mean_delta']:+.4f} | p-valor: {hyp_test['p_value']:.4f} | Conclusão: {hyp_test['conclusion']}")

# 6. Matrizes de Confusão
confusion_matrices = {}
from sklearn.metrics import confusion_matrix
for s_name in feature_sets_names:
    cm = confusion_matrix(all_preds["y_true"], all_preds[s_name])
    confusion_matrices[s_name] = cm.tolist()

confusion_matrices["classes"] = classes

# 7. PERMANOVA: Associação entre Físico-Química e Comunidades Microbianas
print("\n--- 6. PERMANOVA das Variáveis Edáficas sobre 16S e ITS ---")
permanova_edaphic = {}
for var in ["ph_h2o", "organic_carbon", "nitrogen", "phosphorus", "potassium"]:
    valid_idx = manifest_with_soil[var].dropna().index
    if len(valid_idx) > 100:
        # Binarizar por tercis/medianas para teste PERMANOVA
        med = manifest_with_soil.loc[valid_idx, var].median()
        groups = (manifest_with_soil.loc[valid_idx, var] > med).astype(str).tolist()
        
        # Sub-matriz de distâncias
        sub_d_16s = d_16s[np.ix_(valid_idx, valid_idx)]
        sub_d_its = d_its[np.ix_(valid_idx, valid_idx)]
        
        p_16s = run_permanova(sub_d_16s, groups, permutations=499, seed=42)
        p_its = run_permanova(sub_d_its, groups, permutations=499, seed=42)
        
        permanova_edaphic[var] = {
            "16S": {"r2": p_16s["r2"], "p_value": p_16s["p_value"]},
            "ITS": {"r2": p_its["r2"], "p_value": p_its["p_value"]}
        }
        print(f"  {var:15}: 16S R2 = {p_16s['r2']:.4f} (p={p_16s['p_value']:.3f}) | ITS R2 = {p_its['r2']:.4f} (p={p_its['p_value']:.3f})")

# Salvar resultados
(out_dir / "real_spatial_cv_metrics_with_soil.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
(out_dir / "real_confusion_matrices_with_soil.json").write_text(json.dumps(confusion_matrices, indent=2), encoding="utf-8")
(out_dir / "permanova_edaphic_results.json").write_text(json.dumps(permanova_edaphic, indent=2), encoding="utf-8")

print("\nAnálise multimodal concluída e salva com sucesso em results/real_analysis_715/")
