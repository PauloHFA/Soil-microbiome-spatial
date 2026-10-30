"""
run_real_cohort_analysis.py - Executes the complete spatial discrimination pipeline
on the real 715 European paired sites (LUCAS 2018 / Nature Communications).
"""

import json
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.spatial_cv.spatial_partitioner import create_spatial_splits
from src.spatial_cv.leakage_guard import assert_zero_spatial_leakage
from src.evaluation.metrics import compute_classification_metrics, compute_confusion_matrix_summary
from src.evaluation.bootstrap_ci import compute_spatial_bootstrap_ci
from src.evaluation.hypothesis_test import evaluate_multimarker_hypothesis
from src.ecology.multivariate_stats import run_permanova

# 1. Load real data
data_dir = Path("data/real_cohort_715")
out_dir = Path("results/real_analysis_715")
out_dir.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(data_dir / "real_paired_manifest_715.csv")
d_16s = pd.read_csv(data_dir / "real_distance_matrix_16s_715.csv", index_col=0).values
d_its = pd.read_csv(data_dir / "real_distance_matrix_its_715.csv", index_col=0).values
alpha = pd.read_csv(data_dir / "real_alpha_diversity_715.csv", index_col=0)

n_samples = len(manifest)
print(f"Loaded real cohort: {n_samples} paired samples.")

# 2. Compute Multidimensional Scaling (PCoA/MDS embeddings) from distance matrices
# Set A: 16S Bacteria embedding (e.g., 30 principal coordinate dimensions)
print("Computing PCoA embeddings for 16S and ITS distance matrices...")
mds_16s = MDS(n_components=30, dissimilarity="precomputed", random_state=42)
X_16s = mds_16s.fit_transform(d_16s)

mds_its = MDS(n_components=30, dissimilarity="precomputed", random_state=42)
X_its = mds_its.fit_transform(d_its)

# Set C: Integrated Multi-Marker (Early feature fusion)
X_integrated = np.hstack([X_16s, X_its])

# Labels: 6 fine land cover categories
classes = sorted(manifest["fine_land_cover"].unique().tolist())
class_to_code = {c: i for i, c in enumerate(classes)}
y = manifest["fine_land_cover"].map(class_to_code).values

print("\nClass distribution (6 fine land cover categories):")
print(manifest["fine_land_cover"].value_counts())

# 3. Spatial Cross-Validation (5 Folds with Zero Site/Country Leakage)
# Use country / geographic blocking
manifest["spatial_group"] = pd.factorize(manifest["country"])[0]
splits = create_spatial_splits(manifest, n_splits=5, seed=42, group_col="country")

# Verify zero spatial leakage across countries/sites
for tr, te in splits:
    tr_countries = set(manifest.iloc[tr]["country"].dropna())
    te_countries = set(manifest.iloc[te]["country"].dropna())
    assert_zero_spatial_leakage(tr_countries, te_countries)

print("\nZero spatial leakage verified across all 5 cross-validation folds!")

# 4. Train & Evaluate Models Across Folds
feature_sets = {
    "set_a_16s": X_16s,
    "set_b_its": X_its,
    "set_c_integrated": X_integrated
}

fold_scores = {"set_a_16s": [], "set_b_its": [], "set_c_integrated": []}
train_scores = {"set_a_16s": [], "set_b_its": [], "set_c_integrated": []}
all_preds = {"set_a_16s": [], "set_b_its": [], "set_c_integrated": [], "y_true": []}

for fold_idx, (tr, te) in enumerate(splits):
    y_tr, y_te = y[tr], y[te]
    all_preds["y_true"].extend(y_te.tolist())
    
    for s_name, X_full in feature_sets.items():
        X_tr, X_te = X_full[tr], X_full[te]
        
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42 + fold_idx, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        
        pred_tr = clf.predict(X_tr)
        pred_te = clf.predict(X_te)
        
        tr_acc = balanced_accuracy_score(y_tr, pred_tr)
        te_acc = balanced_accuracy_score(y_te, pred_te)
        
        train_scores[s_name].append(tr_acc)
        fold_scores[s_name].append(te_acc)
        all_preds[s_name].extend(pred_te.tolist())

# 5. Scientific Hypothesis Testing (Set C vs Best Single Marker chosen on Training)
scores_a = np.array(fold_scores["set_a_16s"])
scores_b = np.array(fold_scores["set_b_its"])
scores_c = np.array(fold_scores["set_c_integrated"])

hyp = evaluate_multimarker_hypothesis(
    scores_a, scores_b, scores_c,
    training_scores_16s=train_scores["set_a_16s"],
    training_scores_its=train_scores["set_b_its"],
    alternative="greater"
)

# Bootstrap 95% Confidence Intervals
ci_a = compute_spatial_bootstrap_ci(scores_a, n_bootstraps=1000, seed=42)
ci_b = compute_spatial_bootstrap_ci(scores_b, n_bootstraps=1000, seed=42)
ci_c = compute_spatial_bootstrap_ci(scores_c, n_bootstraps=1000, seed=42)

# PERMANOVA on real distance matrices
perm_16s = run_permanova(d_16s, manifest["fine_land_cover"].values, permutations=999, seed=42)
perm_its = run_permanova(d_its, manifest["fine_land_cover"].values, permutations=999, seed=42)

results_summary = {
    "cohort_size": n_samples,
    "target_classes": classes,
    "chance_level": float(1.0 / len(classes)),
    "mean_balanced_accuracy": {
        "set_a_16s": float(np.mean(scores_a)),
        "set_b_its": float(np.mean(scores_b)),
        "set_c_integrated": float(np.mean(scores_c)),
    },
    "fold_balanced_accuracy": fold_scores,
    "confidence_intervals_95": {
        "set_a_16s": ci_a,
        "set_b_its": ci_b,
        "set_c_integrated": ci_c,
    },
    "permanova": {
        "16s": perm_16s,
        "its": perm_its,
    },
    "hypothesis_test_h0": hyp,
}

# Export outputs
with open(out_dir / "real_spatial_cv_metrics.json", "w") as f:
    json.dump(results_summary, f, indent=2)

cm_summary = {
    s_name: compute_confusion_matrix_summary(np.array(all_preds["y_true"]), np.array(all_preds[s_name]), classes)
    for s_name in ["set_a_16s", "set_b_its", "set_c_integrated"]
}
with open(out_dir / "real_confusion_matrices.json", "w") as f:
    json.dump(cm_summary, f, indent=2)

print("\n" + "="*60)
print("REAL COHORT PIPELINE EXECUTION RESULTS (715 SITES)")
print("="*60)
print(f"Mean Balanced Accuracy (16S Bacteria):       {np.mean(scores_a):.4f}")
print(f"Mean Balanced Accuracy (ITS Fungi):          {np.mean(scores_b):.4f}")
print(f"Mean Balanced Accuracy (Integrated 16S+ITS): {np.mean(scores_c):.4f}")
print(f"Delta (Integrated vs Best Single Marker):    {hyp['mean_delta']:+.4f}")
print(f"Hypothesis Test Conclusion: {hyp['conclusion']}")
print(f"PERMANOVA R2: 16S = {perm_16s['r2']:.4f} (p={perm_16s['p_value']:.4f}) | ITS = {perm_its['r2']:.4f} (p={perm_its['p_value']:.4f})")
print("="*60)
