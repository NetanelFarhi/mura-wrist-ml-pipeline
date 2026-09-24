"""Small, dependency-light bootstrap utilities for study-level AUROC."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _as_binary_arrays(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("y_true and y_score must be one-dimensional")
    if len(labels) != len(scores):
        raise ValueError("y_true and y_score must have equal length")
    if len(labels) == 0:
        raise ValueError("At least one study prediction is required")
    if not np.isfinite(scores).all():
        raise ValueError("y_score contains a non-finite value")

    unique = np.unique(labels)
    if not set(unique.tolist()).issubset({0, 1, False, True}):
        raise ValueError("y_true must contain binary labels encoded as 0 and 1")
    labels = labels.astype(int)
    if len(np.unique(labels)) != 2:
        raise ValueError("AUROC requires at least one positive and one negative study")
    return labels, scores


def binary_roc_auc_score(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
) -> float:
    """Calculate AUROC using average ranks, including correct tie handling.

    This avoids making scikit-learn a dependency for the evaluation-only
    workflow and is equivalent to the Mann--Whitney U interpretation of ROC
    area for binary labels.
    """

    labels, scores = _as_binary_arrays(y_true, y_score)
    n_positive = int(labels.sum())
    n_negative = int(len(labels) - n_positive)

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks_sorted = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Ranks are 1-based; all tied observations receive their mean rank.
        ranks_sorted[start:end] = (start + 1 + end) / 2.0
        start = end

    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = ranks_sorted
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2.0) / (
        n_positive * n_negative
    )


def bootstrap_auroc_ci(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    n_bootstraps: int = 2_000,
    confidence_level: float = 0.95,
    random_state: int | None = 42,
    return_samples: bool = False,
) -> dict[str, Any]:
    """Estimate a percentile confidence interval for study-level AUROC.

    A stratified bootstrap resamples positive and negative studies separately.
    That preserves both classes in every replicate, avoiding silently dropped
    samples or undefined AUROC values for small evaluation sets.
    """

    if n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    labels, scores = _as_binary_arrays(y_true, y_score)
    positive_indices = np.flatnonzero(labels == 1)
    negative_indices = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(random_state)
    bootstrap_scores = np.empty(n_bootstraps, dtype=float)

    for index in range(n_bootstraps):
        sampled_indices = np.concatenate(
            (
                rng.choice(positive_indices, size=len(positive_indices), replace=True),
                rng.choice(negative_indices, size=len(negative_indices), replace=True),
            )
        )
        bootstrap_scores[index] = binary_roc_auc_score(
            labels[sampled_indices], scores[sampled_indices]
        )

    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(bootstrap_scores, [alpha, 1.0 - alpha])
    estimate = binary_roc_auc_score(labels, scores)
    result: dict[str, Any] = {
        "metric": "auroc",
        "estimate": float(estimate),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        # Short aliases make the structure convenient in notebooks and tables.
        "lower": float(lower),
        "upper": float(upper),
        "confidence_level": float(confidence_level),
        "n_bootstraps": int(n_bootstraps),
        "random_state": random_state,
        "method": "stratified_study_bootstrap_percentile",
    }
    if return_samples:
        result["samples"] = bootstrap_scores.tolist()
    return result


# Concise names useful in notebooks and external callers.
roc_auc = binary_roc_auc_score
bootstrap_auc_ci = bootstrap_auroc_ci

