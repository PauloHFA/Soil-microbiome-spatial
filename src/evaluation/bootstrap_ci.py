"""
bootstrap_ci.py - Uncertainty intervals, including resampling by spatial unit.
"""

from typing import Dict, Sequence
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def compute_spatial_bootstrap_ci(
    values: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Dict[str, float]:
    """
    Percentile bootstrap of a 1D array of already aggregated scores.

    Prefer compute_spatial_unit_metric_ci when the observations are samples
    nested in sites. This helper remains for fold-level scores.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "median": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}

    if n == 1 or np.all(arr == arr[0]):
        val = float(arr[0])
        return {"mean": val, "median": val, "ci_lower": val, "ci_upper": val, "std": 0.0}

    rng = np.random.RandomState(seed)
    boot_means = np.zeros(n_bootstraps, dtype=np.float64)

    for i in range(n_bootstraps):
        resample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(resample)

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))

    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": float(np.std(boot_means)),
    }


def _score_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    raise ValueError(f"Unsupported metric '{metric}'.")


def compute_spatial_unit_metric_ci(
    y_true: Sequence,
    y_pred: Sequence,
    site_ids: Sequence,
    metric: str = "balanced_accuracy",
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Percentile interval obtained by resampling sites, not rows.

    Samples from the same site stay together. The number of units is the
    number of sites, even when a site contributes several rows.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    sites = np.asarray(site_ids).astype(str)
    if not (len(y_t) == len(y_p) == len(sites)) or len(y_t) == 0:
        raise ValueError("y_true, y_pred and site_ids must be aligned and non-empty.")

    units = np.unique(sites)
    groups = {unit: np.flatnonzero(sites == unit) for unit in units}
    observed = _score_metric(y_t, y_p, metric)

    if len(units) == 1:
        return {
            "point_estimate": float(observed),
            "mean": float(observed),
            "ci_lower": float(observed),
            "ci_upper": float(observed),
            "n_units": 1,
            "n_bootstraps": 0,
            "metric": metric,
            "unit": "site_id",
        }

    rng = np.random.RandomState(seed)
    boots = np.empty(n_bootstraps, dtype=np.float64)
    kept = 0
    for _ in range(n_bootstraps):
        drawn = rng.choice(units, size=len(units), replace=True)
        idx = np.concatenate([groups[unit] for unit in drawn])
        try:
            boots[kept] = _score_metric(y_t[idx], y_p[idx], metric)
            kept += 1
        except ValueError:
            continue

    if kept == 0:
        lower = upper = float(observed)
    else:
        lower = float(np.percentile(boots[:kept], 100.0 * (alpha / 2.0)))
        upper = float(np.percentile(boots[:kept], 100.0 * (1.0 - alpha / 2.0)))

    return {
        "point_estimate": float(observed),
        "mean": float(observed),
        "ci_lower": lower,
        "ci_upper": upper,
        "n_units": int(len(units)),
        "n_bootstraps": int(kept),
        "metric": metric,
        "unit": "site_id",
    }
