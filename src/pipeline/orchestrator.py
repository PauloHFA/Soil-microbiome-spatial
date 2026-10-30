"""
orchestrator.py - Paired spatial comparison required by the revised specification.

The scientific entrypoint does not generate mock data. Sets A, B and C share
the same sites, labels and partitions. The comparator is chosen on training scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json

import numpy as np
import pandas as pd

from src.data.scientific_cohort import (
    LUCAS_STUDY_ACCESSION,
    MANGROVE_STUDY_ACCESSION,
    ScientificCohortError,
    load_scientific_cohort,
)
from src.ecology.alpha_diversity import compute_alpha_diversity_table
from src.ecology.beta_diversity import aitchison_distance_matrix
from src.ecology.multivariate_stats import run_betadisper, run_permanova
from src.evaluation.bootstrap_ci import compute_spatial_unit_metric_ci
from src.evaluation.hypothesis_test import evaluate_multimarker_hypothesis
from src.evaluation.metrics import compute_classification_metrics, compute_confusion_matrix_summary
from src.models.classifiers import get_classifier, train_classifier
from src.models.feature_builder import build_feature_sets, impute_and_scale_inside_fold
from src.spatial_cv.leakage_guard import assert_zero_spatial_leakage
from src.spatial_cv.spatial_partitioner import (
    SpatialValidationError,
    apply_spatial_buffering,
    create_validated_spatial_splits,
    test_categories_represented,
)


SET_A = "set_a_16s"
SET_B = "set_b_its"
SET_C = "set_c_integrated"
MARKER_SETS = (SET_A, SET_B, SET_C)


@dataclass
class PipelineConfig:
    seed: int = 42
    n_splits: int = 5
    classifier: str = "rf"
    target_depth_16s: int = 40109
    target_depth_its: int = 502
    output_dir: str = "results"
    data_dir: str = "data/processed"
    buffer_km: float = 0.0
    n_estimators: int = 150
    n_bootstraps: int = 1000
    label_col: Optional[str] = None
    group_col: str = "site_id"


def _fit_predict(
    x_matrix: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    classifier: str,
    seed: int,
    n_estimators: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    x_train, x_test = impute_and_scale_inside_fold(x_matrix[train_idx], x_matrix[test_idx])
    model = get_classifier(classifier, seed=seed, n_estimators=n_estimators)
    model = train_classifier(model, x_train, y[train_idx])
    predictions = model.predict(x_test)
    return predictions, compute_classification_metrics(y[test_idx], predictions)


def _training_marker_score(
    x_matrix: np.ndarray,
    y: np.ndarray,
    sites: np.ndarray,
    train_idx: np.ndarray,
    config: PipelineConfig,
    fold_seed: int,
) -> Tuple[float, bool]:
    """
    Score one marker using only training rows.

    Inner spatial folds are used when training still has at least two sites.
    Otherwise the in-sample training score is used and the fold is exploratory.
    Test indices are not readable from this function.
    """
    train_sites = sites[train_idx]
    train_labels = y[train_idx]
    n_groups = len(np.unique(train_sites))
    if n_groups >= 2 and len(np.unique(train_labels)) >= 2:
        inner_frame = pd.DataFrame({"site_id": train_sites, "label": train_labels})
        try:
            inner_plan = create_validated_spatial_splits(
                inner_frame,
                n_splits=min(3, n_groups),
                seed=config.seed,
                group_col="site_id",
                label_col="label",
            )
        except SpatialValidationError:
            inner_plan = None
        if inner_plan is not None and inner_plan.splits:
            scores: List[float] = []
            for inner_pos, (inner_train, inner_test) in enumerate(inner_plan.splits):
                predictions, metrics = _fit_predict(
                    x_matrix,
                    y,
                    train_idx[inner_train],
                    train_idx[inner_test],
                    config.classifier,
                    fold_seed + 17 + inner_pos,
                    config.n_estimators,
                )
                del predictions
                scores.append(metrics["balanced_accuracy"])
            return float(np.mean(scores)), inner_plan.exploratory

    predictions, metrics = _fit_predict(
        x_matrix,
        y,
        train_idx,
        train_idx,
        config.classifier,
        fold_seed,
        config.n_estimators,
    )
    del predictions
    return metrics["balanced_accuracy"], True


def _ecology_block(
    counts: np.ndarray,
    labels: Sequence[Any],
    seed: int,
) -> Dict[str, Any]:
    distance = aitchison_distance_matrix(counts)
    permanova = run_permanova(distance, labels, seed=seed)
    dispersion = run_betadisper(distance, labels)
    return {
        "distance": "aitchison",
        "permanova": permanova,
        "multivariate_dispersion": dispersion,
        "dispersion_limits_permanova": not bool(dispersion.get("homogeneous", True)),
        "role": "descriptive; not a substitute for spatial holdout",
    }


def run_paired_comparison(
    manifest: pd.DataFrame,
    counts_16s: np.ndarray,
    counts_its: np.ndarray,
    config: Optional[PipelineConfig] = None,
    label_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run A, B and C on one paired cohort and one spatial partition plan.

    Caller must already have refused cross-study and mangrove rows.
    """
    cfg = config or PipelineConfig()
    if len(manifest) != len(counts_16s) or len(manifest) != len(counts_its):
        raise ScientificCohortError("Manifest and count matrices are not row-aligned.")
    if cfg.group_col not in manifest.columns:
        raise SpatialValidationError(
            f"Column '{cfg.group_col}' is required. Random split is not available."
        )

    resolved_label = label_col or cfg.label_col
    if resolved_label is None:
        for candidate in ("fine_land_cover", "macro_land_cover", "land_cover"):
            if candidate in manifest.columns:
                resolved_label = candidate
                break
    if resolved_label is None or resolved_label not in manifest.columns:
        raise ScientificCohortError("No environmental category column is available.")

    labels_raw = manifest[resolved_label].astype(str).to_numpy()
    class_labels = sorted(set(labels_raw))
    label_to_code = {label: idx for idx, label in enumerate(class_labels)}
    y = np.asarray([label_to_code[label] for label in labels_raw], dtype=int)
    sites = manifest[cfg.group_col].astype(str).to_numpy()
    chance = 1.0 / float(len(class_labels))

    plan = create_validated_spatial_splits(
        manifest,
        n_splits=cfg.n_splits,
        seed=cfg.seed,
        group_col=cfg.group_col,
        label_col=resolved_label,
    )

    feature_sets = build_feature_sets(counts_16s, counts_its)
    fold_scores = {name: [] for name in MARKER_SETS}
    fold_macro_f1 = {name: [] for name in MARKER_SETS}
    train_scores = {SET_A: [], SET_B: []}
    oof_true: List[int] = []
    oof_sites: List[str] = []
    oof_pred = {name: [] for name in MARKER_SETS}
    fold_audit: List[Dict[str, Any]] = []
    exploratory = bool(plan.exploratory)
    used_splits: List[Tuple[np.ndarray, np.ndarray]] = []

    lats = manifest["latitude"].to_numpy(dtype=float) if "latitude" in manifest.columns else None
    lons = manifest["longitude"].to_numpy(dtype=float) if "longitude" in manifest.columns else None

    for fold_idx, (train_idx, test_idx) in enumerate(plan.splits):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)
        if cfg.buffer_km > 0:
            if lats is None or lons is None:
                raise SpatialValidationError("Buffer requested without coordinates.")
            train_idx = apply_spatial_buffering(
                train_idx, test_idx, lats, lons, labels_raw, buffer_km=cfg.buffer_km
            )
            if len(train_idx) == 0 or not test_categories_represented(labels_raw[train_idx], labels_raw[test_idx]):
                exploratory = True
                continue
        assert_zero_spatial_leakage(set(sites[train_idx]), set(sites[test_idx]))
        if not test_categories_represented(y[train_idx], y[test_idx]):
            exploratory = True
            continue

        train_a, inner_flag_a = _training_marker_score(
            feature_sets[SET_A], y, sites, train_idx, cfg, cfg.seed + fold_idx
        )
        train_b, inner_flag_b = _training_marker_score(
            feature_sets[SET_B], y, sites, train_idx, cfg, cfg.seed + 100 + fold_idx
        )
        exploratory = exploratory or inner_flag_a or inner_flag_b
        train_scores[SET_A].append(train_a)
        train_scores[SET_B].append(train_b)
        selected = SET_A if train_a >= train_b else SET_B

        fold_metrics: Dict[str, Dict[str, float]] = {}
        for set_name in MARKER_SETS:
            predictions, metrics = _fit_predict(
                feature_sets[set_name],
                y,
                train_idx,
                test_idx,
                cfg.classifier,
                cfg.seed + fold_idx,
                cfg.n_estimators,
            )
            fold_scores[set_name].append(metrics["balanced_accuracy"])
            fold_macro_f1[set_name].append(metrics["macro_f1"])
            fold_metrics[set_name] = metrics
            oof_pred[set_name].extend(predictions.tolist())
        oof_true.extend(y[test_idx].tolist())
        oof_sites.extend(sites[test_idx].tolist())
        used_splits.append((train_idx, test_idx))
        fold_audit.append({
            "fold": fold_idx,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "selected_marker_on_training": "16S" if selected == SET_A else "ITS",
            "training_balanced_accuracy_16s": train_a,
            "training_balanced_accuracy_its": train_b,
            "tie_break": "predeclared_16s" if train_a == train_b else "training_score",
        })

    if not used_splits:
        raise SpatialValidationError(
            "Buffering or category coverage removed every spatial fold. "
            "Random split and buffer-point rescue are not used."
        )

    hypothesis = evaluate_multimarker_hypothesis(
        np.asarray(fold_scores[SET_A]),
        np.asarray(fold_scores[SET_B]),
        np.asarray(fold_scores[SET_C]),
        training_scores_16s=np.asarray(train_scores[SET_A]),
        training_scores_its=np.asarray(train_scores[SET_B]),
    )
    y_oof = np.asarray(oof_true)
    site_oof = np.asarray(oof_sites)
    intervals = {}
    matrices = {}
    for set_name in MARKER_SETS:
        pred = np.asarray(oof_pred[set_name])
        intervals[set_name] = {
            "balanced_accuracy": compute_spatial_unit_metric_ci(
                y_oof, pred, site_oof, metric="balanced_accuracy",
                n_bootstraps=cfg.n_bootstraps, seed=cfg.seed,
            ),
            "macro_f1": compute_spatial_unit_metric_ci(
                y_oof, pred, site_oof, metric="macro_f1",
                n_bootstraps=cfg.n_bootstraps, seed=cfg.seed,
            ),
        }
        matrices[set_name] = compute_confusion_matrix_summary(
            y_oof, pred, class_labels, label_ids=list(range(len(class_labels)))
        )

    return {
        "classifier": cfg.classifier,
        "seed": cfg.seed,
        "target_column": resolved_label,
        "target_classes": class_labels,
        "chance_balanced_accuracy": chance,
        "n_splits_requested": plan.n_splits_requested,
        "n_splits_used": len(used_splits),
        "exploratory": exploratory,
        "dropped_folds": plan.dropped_folds + (len(plan.splits) - len(used_splits)),
        "spatial_validation": "site_or_block_holdout",
        "random_split_used": False,
        "interpretation": (
            "Environmental category discrimination under spatial holdout. "
            "Not origin attribution."
        ),
        "mean_balanced_accuracy": {
            name: float(np.mean(fold_scores[name])) for name in MARKER_SETS
        },
        "mean_macro_f1": {
            name: float(np.mean(fold_macro_f1[name])) for name in MARKER_SETS
        },
        "fold_balanced_accuracy": fold_scores,
        "fold_macro_f1": fold_macro_f1,
        "confidence_intervals_by_site": intervals,
        "confusion_matrices": matrices,
        "hypothesis_test_h0": hypothesis,
        "fold_audit": fold_audit,
        "permanova": {
            "16s": _ecology_block(counts_16s, labels_raw, cfg.seed),
            "its": _ecology_block(counts_its, labels_raw, cfg.seed + 1),
        },
    }


def run_pipeline(config: Optional[PipelineConfig] = None) -> Dict[str, Any]:
    """
    Scientific entrypoint - follows exactly the revised post-doc specification.

    Loads only real LUCAS/ESDAC paired cohort (PRJNA952168).
    Never uses mock data for the main scientific comparison.
    Enforces same samples + same spatial partitions for Sets A, B and C.
    """
    cfg = config or PipelineConfig()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load only real scientific cohort (rejects mock and mangrove for main track)
    manifest, counts_16s, counts_its, label_col = load_scientific_cohort(
        cfg.data_dir,
        preferred_label=cfg.label_col,
    )
    if cfg.label_col is None:
        cfg.label_col = label_col

    # Alpha diversity (required by post-doc report)
    sample_ids = manifest["sample_id"].astype(str).tolist()
    alpha_16s = compute_alpha_diversity_table(counts_16s, sample_ids, prefix="16s")
    alpha_its = compute_alpha_diversity_table(counts_its, sample_ids, prefix="its")
    alpha_df = pd.merge(alpha_16s, alpha_its, on="sample_id")
    alpha_df.to_csv(out_dir / "alpha_diversity.csv", index=False)

    # Core scientific comparison (A vs B vs C on identical spatial folds)
    summary = run_paired_comparison(
        manifest, counts_16s, counts_its, config=cfg, label_col=label_col
    )
    summary["study_accession"] = LUCAS_STUDY_ACCESSION
    summary["mangrove_in_set_c"] = False
    # Software can follow the analytical design and still not be an empirical
    # test of the revised postdoctoral project. Compliance is not inferred
    # from a study accession stamp.
    summary["postdoc_revised_spec_compliant"] = False
    summary["compliance_note"] = (
        "Analytical design only. Empirical compliance requires audited public "
        "LUCAS/ESDAC files, not a synthetic cohort stamped as PRJNA952168."
    )

    # Export all artifacts required by the revised project
    exportable = {key: value for key, value in summary.items() if key != "confusion_matrices"}
    with open(out_dir / "spatial_cv_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(exportable, handle, indent=2, default=str)

    with open(out_dir / "confusion_matrices.json", "w", encoding="utf-8") as handle:
        json.dump(summary["confusion_matrices"], handle, indent=2, default=str)

    # Also export a human-readable summary
    with open(out_dir / "hypothesis_summary.txt", "w", encoding="utf-8") as f:
        h = summary["hypothesis_test_h0"]
        f.write("=== POST-DOC REVISED SPECIFICATION REPORT ===\n\n")
        f.write(f"Classifier: {summary['classifier']}\n")
        f.write(f"Seed: {summary['seed']}\n")
        f.write(f"Spatial splits used: {summary['n_splits_used']}/{summary['n_splits_requested']}\n\n")
        f.write(f"Mean Balanced Accuracy (16S):     {summary['mean_balanced_accuracy']['set_a_16s']:.4f}\n")
        f.write(f"Mean Balanced Accuracy (ITS):     {summary['mean_balanced_accuracy']['set_b_its']:.4f}\n")
        f.write(f"Mean Balanced Accuracy (Integrated): {summary['mean_balanced_accuracy']['set_c_integrated']:.4f}\n\n")
        f.write(f"H0 (Integrated not superior): {h['conclusion']}\n")
        f.write(f"p-value: {h.get('p_value', 'N/A')}\n")
        f.write(f"Interpretation: {summary.get('interpretation', '')}\n")

    print(f"Pipeline completed. Results written to {out_dir}")
    return summary


def run_mangrove_16s_track(
    manifest: pd.DataFrame,
    counts_16s: np.ndarray,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """
    Isolated 16S track for PRJNA283936.

    There is no set B and no set C. Calling this with ITS runs is an error.
    """
    cfg = config or PipelineConfig()
    if manifest is None or manifest.empty:
        raise ScientificCohortError("Mangrove 16S track requires its own manifest.")
    if "study_accession" in manifest.columns:
        studies = set(manifest["study_accession"].astype(str))
    elif "bioproject" in manifest.columns:
        studies = set(manifest["bioproject"].astype(str))
    else:
        raise ScientificCohortError("Mangrove track requires study_accession or bioproject.")
    if studies != {MANGROVE_STUDY_ACCESSION}:
        raise ScientificCohortError(
            f"Isolated 16S track accepts only {MANGROVE_STUDY_ACCESSION}. Found {sorted(studies)}."
        )
    for column in ("sra_its_run", "its_run", "run_its"):
        if column in manifest.columns and manifest[column].notna().any():
            nonempty = manifest[column].astype(str).str.strip()
            if nonempty.replace("nan", "").ne("").any():
                raise ScientificCohortError(
                    "PRJNA283936 cannot be paired with ITS and cannot enter set C."
                )
    if "marker" in manifest.columns:
        markers = set(manifest["marker"].dropna().astype(str).str.upper())
        if markers - {"16S"}:
            raise ScientificCohortError(
                f"PRJNA283936 track refuses markers {sorted(markers)} and does not build set C."
            )

    label_col = "estuary" if "estuary" in manifest.columns else cfg.label_col
    if label_col is None or label_col not in manifest.columns:
        raise ScientificCohortError("Mangrove track needs a category column, usually estuary.")
    group_col = cfg.group_col if cfg.group_col in manifest.columns else "estuary"
    if group_col not in manifest.columns:
        raise SpatialValidationError("Mangrove track has no spatial group column.")

    local = manifest.copy()
    if group_col != cfg.group_col:
        local[cfg.group_col] = local[group_col]
    labels_raw = local[label_col].astype(str).to_numpy()
    class_labels = sorted(set(labels_raw))
    codes = {label: idx for idx, label in enumerate(class_labels)}
    y = np.asarray([codes[label] for label in labels_raw], dtype=int)
    sites = local[cfg.group_col].astype(str).to_numpy()
    plan = create_validated_spatial_splits(
        local,
        n_splits=min(cfg.n_splits, max(2, sites_nunique(sites))),
        seed=cfg.seed,
        group_col=cfg.group_col,
        label_col=label_col,
    )
    from src.ecology.compositional import clr_transform

    features = clr_transform(counts_16s)
    scores: List[float] = []
    macro: List[float] = []
    oof_true: List[int] = []
    oof_pred: List[int] = []
    oof_sites: List[str] = []
    for fold_idx, (train_idx, test_idx) in enumerate(plan.splits):
        assert_zero_spatial_leakage(set(sites[train_idx]), set(sites[test_idx]))
        predictions, metrics = _fit_predict(
            features, y, train_idx, test_idx, cfg.classifier, cfg.seed + fold_idx, cfg.n_estimators
        )
        scores.append(metrics["balanced_accuracy"])
        macro.append(metrics["macro_f1"])
        oof_true.extend(y[test_idx].tolist())
        oof_pred.extend(predictions.tolist())
        oof_sites.extend(sites[test_idx].tolist())

    y_oof = np.asarray(oof_true)
    pred = np.asarray(oof_pred)
    result = {
        "study_accession": MANGROVE_STUDY_ACCESSION,
        "track": "16S_only",
        "integration_tested": False,
        "set_c_integrated": None,
        "mangrove_in_set_c": False,
        "exploratory": True,
        "interpretation": (
            "Complementary bacterial mangrove track. Not an integration test and "
            "not a reference for Paraíba soils."
        ),
        "mean_balanced_accuracy": {"set_a_16s": float(np.mean(scores)) if scores else 0.0},
        "mean_macro_f1": {"set_a_16s": float(np.mean(macro)) if macro else 0.0},
        "confidence_intervals_by_site": {
            "set_a_16s": compute_spatial_unit_metric_ci(
                y_oof, pred, np.asarray(oof_sites), n_bootstraps=cfg.n_bootstraps, seed=cfg.seed
            )
        },
        "confusion_matrices": {
            "set_a_16s": compute_confusion_matrix_summary(
                y_oof, pred, class_labels, label_ids=list(range(len(class_labels)))
            )
        },
        "permanova": {"16s": _ecology_block(counts_16s, labels_raw, cfg.seed)},
    }
    if SET_C in result.get("mean_balanced_accuracy", {}):
        raise ScientificCohortError("Mangrove track leaked into set C.")
    return result


def sites_nunique(sites: np.ndarray) -> int:
    return int(len(np.unique(sites)))
