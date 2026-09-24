"""Study-level binary classification metrics and validation-only selection tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


ThresholdObjective = Literal["youden_j", "f1", "balanced_accuracy"]


def _validate_labels_and_scores(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    y_score = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(y_true) == 0:
        raise ValueError("Metrics require at least one study.")
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"labels and probabilities must have equal shape; got {y_true.shape} and {y_score.shape}."
        )
    labels_seen = set(y_true.tolist())
    if not labels_seen.issubset({0, 1}):
        raise ValueError(f"Expected binary labels 0/1, found {sorted(labels_seen)}.")
    if not np.all(np.isfinite(y_score)):
        raise ValueError("Probabilities contain NaN or infinity.")
    if np.any((y_score < 0.0) | (y_score > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1].")
    return y_true, y_score


def safe_auroc(labels: Sequence[int] | np.ndarray, probabilities: Sequence[float] | np.ndarray) -> float | None:
    """Return AUROC, or ``None`` if a split contains only one class."""

    y_true, y_score = _validate_labels_and_scores(labels, probabilities)
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(labels: Sequence[int] | np.ndarray, probabilities: Sequence[float] | np.ndarray) -> float | None:
    """Return average precision (AUPRC), or ``None`` for a one-class split."""

    y_true, y_score = _validate_labels_and_scores(labels, probabilities)
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


@dataclass(frozen=True)
class BinaryStudyMetrics:
    """JSON-friendly study-level metrics at one fixed operating threshold."""

    threshold: float
    n_studies: int
    n_normal: int
    n_abnormal: int
    auroc: float | None
    auprc: float | None
    sensitivity: float | None
    specificity: float | None
    f1: float
    false_negative_rate: float | None
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    @property
    def recall(self) -> float | None:
        """Sensitivity alias used in the Phase 1 comparison table."""

        return self.sensitivity

    @property
    def confusion_matrix(self) -> list[list[int]]:
        return [
            [self.true_negative, self.false_positive],
            [self.false_negative, self.true_positive],
        ]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["recall"] = self.recall
        payload["confusion_matrix"] = self.confusion_matrix
        return payload


def compute_binary_study_metrics(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> BinaryStudyMetrics:
    """Compute all required Phase 1 metrics at a fixed study-level threshold."""

    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}.")
    y_true, y_score = _validate_labels_and_scores(labels, probabilities)
    y_pred = (y_score >= float(threshold)).astype(np.int64)

    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))

    sensitivity_denominator = true_positive + false_negative
    specificity_denominator = true_negative + false_positive
    sensitivity = (
        float(true_positive / sensitivity_denominator) if sensitivity_denominator else None
    )
    specificity = float(true_negative / specificity_denominator) if specificity_denominator else None
    false_negative_rate = (
        float(false_negative / sensitivity_denominator) if sensitivity_denominator else None
    )
    return BinaryStudyMetrics(
        threshold=float(threshold),
        n_studies=int(len(y_true)),
        n_normal=int(np.sum(y_true == 0)),
        n_abnormal=int(np.sum(y_true == 1)),
        auroc=safe_auroc(y_true, y_score),
        auprc=safe_auprc(y_true, y_score),
        sensitivity=sensitivity,
        specificity=specificity,
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        false_negative_rate=false_negative_rate,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        true_positive=true_positive,
    )


@dataclass(frozen=True)
class ThresholdSelection:
    """A threshold selected exclusively from a supplied validation prediction set."""

    threshold: float
    objective: ThresholdObjective
    objective_value: float
    candidates_evaluated: int
    tie_break: str = "closest_to_0.5_then_lower_threshold"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _objective_value(metrics: BinaryStudyMetrics, objective: ThresholdObjective) -> float:
    if objective == "youden_j":
        if metrics.sensitivity is None or metrics.specificity is None:
            return float("-inf")
        return metrics.sensitivity + metrics.specificity - 1.0
    if objective == "balanced_accuracy":
        if metrics.sensitivity is None or metrics.specificity is None:
            return float("-inf")
        return (metrics.sensitivity + metrics.specificity) / 2.0
    if objective == "f1":
        return metrics.f1
    raise ValueError(f"Unsupported threshold objective {objective!r}.")


def select_operating_threshold_on_validation(
    validation_labels: Sequence[int] | np.ndarray,
    validation_probabilities: Sequence[float] | np.ndarray,
    *,
    objective: ThresholdObjective = "youden_j",
) -> ThresholdSelection:
    """Select a reproducible operating threshold using validation studies only.

    Defaulting to Youden's J avoids silently privileging either sensitivity or
    specificity.  It is not a clinical operating point; a deployment-specific
    trade-off requires explicit clinical justification.  Ties are resolved
    deterministically by proximity to 0.5, then a lower threshold.
    """

    y_true, y_score = _validate_labels_and_scores(validation_labels, validation_probabilities)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Validation threshold selection requires both NORMAL and ABNORMAL studies.")
    if objective not in ("youden_j", "f1", "balanced_accuracy"):
        raise ValueError(f"Unsupported threshold objective {objective!r}.")

    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], y_score))).astype(np.float64)
    best_threshold: float | None = None
    best_value = float("-inf")
    epsilon = 1e-12
    for candidate in candidates.tolist():
        candidate_metrics = compute_binary_study_metrics(y_true, y_score, threshold=float(candidate))
        value = _objective_value(candidate_metrics, objective)
        if value > best_value + epsilon:
            best_value, best_threshold = value, float(candidate)
            continue
        if abs(value - best_value) <= epsilon and best_threshold is not None:
            old_key = (abs(best_threshold - 0.5), best_threshold)
            new_key = (abs(float(candidate) - 0.5), float(candidate))
            if new_key < old_key:
                best_threshold = float(candidate)

    if best_threshold is None or not np.isfinite(best_value):  # defensive; labels were checked above
        raise RuntimeError("Unable to select a threshold from validation predictions.")
    return ThresholdSelection(
        threshold=best_threshold,
        objective=objective,
        objective_value=float(best_value),
        candidates_evaluated=int(len(candidates)),
    )


def metrics_at_default_and_selected_threshold(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    selected_threshold: float,
) -> dict[str, dict[str, object]]:
    """Return the required 0.5 and frozen-operating-point metric reports."""

    return {
        "threshold_0_5": compute_binary_study_metrics(labels, probabilities, threshold=0.5).to_dict(),
        "threshold_selected_on_validation": compute_binary_study_metrics(
            labels, probabilities, threshold=selected_threshold
        ).to_dict(),
    }


def bootstrap_auroc_ci(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int | None]:
    """Compute a deterministic nonparametric bootstrap confidence interval for AUROC.

    Resampling happens at the supplied row level.  Call it only after ensuring
    each row is one study (as this module's public callers do).
    """

    y_true, y_score = _validate_labels_and_scores(labels, probabilities)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1.")
    estimate = safe_auroc(y_true, y_score)
    if estimate is None:
        return {
            "estimate": None,
            "lower": None,
            "upper": None,
            "n_bootstrap": int(n_bootstrap),
            "valid_resamples": 0,
            "confidence": float(confidence),
            "seed": int(seed),
        }

    rng = np.random.default_rng(seed)
    bootstrap_scores: list[float] = []
    indices = np.arange(len(y_true))
    for _ in range(n_bootstrap):
        sampled_indices = rng.choice(indices, size=len(indices), replace=True)
        sampled_labels = y_true[sampled_indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        bootstrap_scores.append(float(roc_auc_score(sampled_labels, y_score[sampled_indices])))
    if not bootstrap_scores:
        lower = upper = None
    else:
        alpha = (1.0 - confidence) / 2.0
        lower, upper = np.quantile(bootstrap_scores, [alpha, 1.0 - alpha]).tolist()
    return {
        "estimate": float(estimate),
        "lower": float(lower) if lower is not None else None,
        "upper": float(upper) if upper is not None else None,
        "n_bootstrap": int(n_bootstrap),
        "valid_resamples": int(len(bootstrap_scores)),
        "confidence": float(confidence),
        "seed": int(seed),
    }


__all__ = [
    "BinaryStudyMetrics",
    "ThresholdSelection",
    "bootstrap_auroc_ci",
    "compute_binary_study_metrics",
    "metrics_at_default_and_selected_threshold",
    "safe_auprc",
    "safe_auroc",
    "select_operating_threshold_on_validation",
]
