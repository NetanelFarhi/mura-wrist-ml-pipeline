"""Validation-only calibration and operating-policy utilities for Phase 2A."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def require_validation_predictions(path: str | Path) -> Path:
    """Reject a test artifact before any Phase 2A analysis can read it."""
    resolved = Path(path)
    if "test" in resolved.name.lower() or "test" in str(resolved.parent).lower():
        raise ValueError("Phase 2A accepts validation predictions only; test predictions are forbidden.")
    return resolved


def load_validation_predictions(path: str | Path) -> pd.DataFrame:
    path = require_validation_predictions(path)
    frame = pd.read_csv(path)
    required = {"patient_id", "study_id", "label", "num_views", "abnormal_probability"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Validation predictions missing columns: {sorted(missing)}")
    if frame.study_id.duplicated().any():
        raise ValueError("Phase 2A validation predictions must have one row per study.")
    if not set(frame.label.unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary.")
    if not frame.abnormal_probability.between(0, 1).all():
        raise ValueError("Probabilities must lie in [0, 1].")
    return frame.copy()


def validate_per_view_export(frame: pd.DataFrame) -> None:
    """Require one real image-path row per exported view, never padded slots."""
    required = {"patient_id", "study_id", "image_path", "image_view_id", "study_label", "view_probability", "num_views"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Per-view export missing columns: {sorted(missing)}")
    if frame.image_path.isna().any() or (frame.image_path.astype(str).str.strip() == "").any():
        raise ValueError("Padded/nonexistent views cannot be exported as real predictions.")
    counts = frame.groupby("study_id").size()
    declared = frame.groupby("study_id").num_views.first()
    if not counts.eq(declared).all() or frame.duplicated(["study_id", "image_path"]).any():
        raise ValueError("Per-view export does not preserve exact real-view study identity.")


def operating_metrics(labels: Iterable[int], probabilities: Iterable[float], threshold: float) -> dict[str, float | int]:
    y = np.asarray(list(labels), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    pred = p >= threshold
    tn = int(np.sum((y == 0) & ~pred)); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred)); tp = int(np.sum((y == 1) & pred))
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {"threshold": float(threshold), "sensitivity": sensitivity, "specificity": specificity,
            "precision": precision, "f1": f1, "false_positive_count": fp, "false_negative_count": fn,
            "true_positive_count": tp, "true_negative_count": tn, "flagged_abnormal_count": int(pred.sum()),
            "flagged_abnormal_percent": float(pred.mean() * 100.0), "n_studies": int(y.size)}


def select_sensitivity_constrained_threshold(labels: Iterable[int], probabilities: Iterable[float], target: float) -> dict[str, float | int]:
    """Highest score threshold with validation sensitivity at least ``target``."""
    if not 0 < target <= 1:
        raise ValueError("target must be in (0, 1].")
    y = np.asarray(list(labels), dtype=int); p = np.asarray(list(probabilities), dtype=float)
    if set(y) != {0, 1}:
        raise ValueError("Both classes are required for sensitivity-constrained selection.")
    candidates = np.unique(np.r_[0.0, 1.0, p])
    feasible = [operating_metrics(y, p, float(t)) for t in candidates]
    feasible = [row for row in feasible if float(row["sensitivity"]) >= target]
    if not feasible:
        raise ValueError("No threshold meets the sensitivity target.")
    # Maximize threshold first. At ties, specificity is reported but cannot be worsened by this choice.
    return sorted(feasible, key=lambda row: (float(row["threshold"]), float(row["specificity"])), reverse=True)[0]


def expected_calibration_error(labels: Iterable[int], probabilities: Iterable[float], n_bins: int = 10) -> float:
    y = np.asarray(list(labels), dtype=float); p = np.asarray(list(probabilities), dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(p, bins[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = index == b
        if mask.any():
            ece += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def calibration_metrics(labels: Iterable[int], probabilities: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(labels), dtype=int); p = np.asarray(list(probabilities), dtype=float)
    q = np.clip(p, 1e-7, 1 - 1e-7)
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "nll": float(log_loss(y, q, labels=[0, 1])),
            "ece_10_bins": expected_calibration_error(y, p, 10)}


def apply_temperature(probabilities: Iterable[float], temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    p = np.clip(np.asarray(list(probabilities), dtype=float), 1e-7, 1 - 1e-7)
    logits = np.log(p / (1 - p))
    return 1 / (1 + np.exp(-logits / temperature))


def fit_temperature(labels: Iterable[int], probabilities: Iterable[float]) -> float:
    y = np.asarray(list(labels), dtype=int); p = np.asarray(list(probabilities), dtype=float)
    def objective(log_temp: float) -> float:
        scaled = apply_temperature(p, float(np.exp(log_temp)))
        return float(log_loss(y, np.clip(scaled, 1e-7, 1 - 1e-7), labels=[0, 1]))
    result = minimize_scalar(objective, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature optimization failed")
    return float(np.exp(result.x))


def cross_fitted_temperature(labels: Iterable[int], probabilities: Iterable[float], seed: int = 23, n_splits: int = 5) -> tuple[np.ndarray, list[float]]:
    y = np.asarray(list(labels), dtype=int); p = np.asarray(list(probabilities), dtype=float)
    if min(np.bincount(y)) < n_splits:
        raise ValueError("Insufficient class support for cross-fitted calibration.")
    out = np.empty_like(p); temperatures: list[float] = []
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit_idx, eval_idx in folds.split(p, y):
        t = fit_temperature(y[fit_idx], p[fit_idx]); temperatures.append(t)
        out[eval_idx] = apply_temperature(p[eval_idx], t)
    return out, temperatures


def bootstrap_policy_ci(labels: Iterable[int], probabilities: Iterable[float], threshold: float, seed: int = 23, n_bootstrap: int = 2000) -> dict[str, float | int]:
    y = np.asarray(list(labels), dtype=int); p = np.asarray(list(probabilities), dtype=float)
    rng = np.random.default_rng(seed); neg=np.flatnonzero(y == 0); pos=np.flatnonzero(y == 1)
    sensitivity=[]; specificity=[]
    for _ in range(n_bootstrap):
        idx = np.r_[rng.choice(neg, size=neg.size, replace=True), rng.choice(pos, size=pos.size, replace=True)]
        row=operating_metrics(y[idx], p[idx], threshold); sensitivity.append(float(row["sensitivity"])); specificity.append(float(row["specificity"]))
    return {"n_bootstrap": n_bootstrap, "seed": seed,
            "sensitivity_ci_lower": float(np.quantile(sensitivity, .025)), "sensitivity_ci_upper": float(np.quantile(sensitivity, .975)),
            "specificity_ci_lower": float(np.quantile(specificity, .025)), "specificity_ci_upper": float(np.quantile(specificity, .975))}
