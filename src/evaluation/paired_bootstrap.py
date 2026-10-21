"""Paired patient-cluster bootstrap for development-only model comparisons."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.phase2a import select_sensitivity_constrained_threshold


def _as_arrays(
    labels: Iterable[int],
    baseline_probabilities: Iterable[float],
    candidate_probabilities: Iterable[float],
    patient_ids: Iterable[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(list(labels), dtype=int)
    baseline = np.asarray(list(baseline_probabilities), dtype=float)
    candidate = np.asarray(list(candidate_probabilities), dtype=float)
    patients = np.asarray(list(patient_ids), dtype=str)
    if not (len(y) == len(baseline) == len(candidate) == len(patients)):
        raise ValueError("Labels, paired probabilities, and patient IDs must have equal length.")
    if len(y) == 0 or not set(y).issubset({0, 1}) or len(np.unique(y)) < 2:
        raise ValueError("Paired bootstrap requires non-empty binary data with both classes.")
    if not (np.isfinite(baseline).all() and np.isfinite(candidate).all()):
        raise ValueError("Paired probabilities must be finite.")
    if np.any((baseline < 0) | (baseline > 1)) or np.any((candidate < 0) | (candidate > 1)):
        raise ValueError("Paired probabilities must lie in [0, 1].")
    return y, baseline, candidate, patients


def _specificity_at_sensitivity(labels: np.ndarray, probabilities: np.ndarray, target: float) -> float:
    return float(select_sensitivity_constrained_threshold(labels, probabilities, target)["specificity"])


def paired_patient_bootstrap(
    labels: Sequence[int] | np.ndarray,
    baseline_probabilities: Sequence[float] | np.ndarray,
    candidate_probabilities: Sequence[float] | np.ndarray,
    patient_ids: Sequence[str] | np.ndarray,
    *,
    sensitivity_targets: Sequence[float] = (0.95,),
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate candidate-minus-baseline uncertainty using paired patient resampling.

    Each bootstrap replicate draws validation patients with replacement, keeping
    all of each sampled patient's studies.  Thresholds inside a replicate are
    recomputed only to estimate uncertainty of the fixed *policy functional*;
    this does not choose a deployment threshold or consult any test set.
    """

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least one.")
    if any(not 0 < float(target) <= 1 for target in sensitivity_targets):
        raise ValueError("Sensitivity targets must lie in (0, 1].")
    y, baseline, candidate, patients = _as_arrays(
        labels, baseline_probabilities, candidate_probabilities, patient_ids
    )
    patient_to_indices: Mapping[str, np.ndarray] = {
        patient: np.flatnonzero(patients == patient) for patient in np.unique(patients)
    }
    unique_patients = np.asarray(sorted(patient_to_indices), dtype=str)
    if len(unique_patients) < 2:
        raise ValueError("Patient-cluster bootstrap needs at least two patients.")

    def measures(indices: np.ndarray) -> dict[str, float]:
        sampled_y = y[indices]
        sampled_baseline = baseline[indices]
        sampled_candidate = candidate[indices]
        if len(np.unique(sampled_y)) < 2:
            raise ValueError("Bootstrap replicate contains one class.")
        values = {
            "AUROC": float(roc_auc_score(sampled_y, sampled_candidate) - roc_auc_score(sampled_y, sampled_baseline)),
            "AUPRC": float(
                average_precision_score(sampled_y, sampled_candidate)
                - average_precision_score(sampled_y, sampled_baseline)
            ),
        }
        for target in sensitivity_targets:
            candidate_specificity = _specificity_at_sensitivity(sampled_y, sampled_candidate, float(target))
            baseline_specificity = _specificity_at_sensitivity(sampled_y, sampled_baseline, float(target))
            values[f"Specificity@{float(target):.0%}"] = candidate_specificity - baseline_specificity
        return values

    point = measures(np.arange(len(y)))
    samples: dict[str, list[float]] = defaultdict(list)
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        chosen = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        indices = np.concatenate([patient_to_indices[patient] for patient in chosen])
        try:
            values = measures(indices)
        except ValueError:
            continue
        for name, value in values.items():
            samples[name].append(float(value))

    rows: list[dict[str, object]] = []
    for name, point_estimate in point.items():
        values = np.asarray(samples[name], dtype=float)
        if values.size == 0:
            raise RuntimeError(f"No valid bootstrap replicates for {name}.")
        lower, upper = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "metric": name,
                "difference_candidate_minus_baseline": float(point_estimate),
                "ci_lower_95": float(lower),
                "ci_upper_95": float(upper),
                "n_bootstrap_requested": int(n_bootstrap),
                "n_bootstrap_valid": int(values.size),
                "seed": int(seed),
                "resampling_unit": "patient cluster",
            }
        )
    return pd.DataFrame(rows)


__all__ = ["paired_patient_bootstrap"]
