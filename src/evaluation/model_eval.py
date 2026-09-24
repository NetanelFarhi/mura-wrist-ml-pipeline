"""Study-level metrics, validation-only threshold selection, and persistence.

All public evaluators in this module expect *one probability per study*.
``aggregate_view_predictions`` is provided only to convert per-view outputs to
that unit before evaluation; it is not an image-level scoring shortcut.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bootstrap import binary_roc_auc_score, bootstrap_auroc_ci


def _normalise_split(value: object) -> str:
    normalised = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return {"training": "train", "val": "validation", "valid": "validation"}.get(
        normalised, normalised
    )


def _binary_labels(values: Sequence[object] | np.ndarray | pd.Series) -> np.ndarray:
    series = pd.Series(values)
    if series.isna().any():
        raise ValueError("Study labels contain a missing value")

    # Native Phase-1 manifests use 0 (NORMAL) and 1 (ABNORMAL), while the
    # string forms make the evaluator usable with a plainly labelled export.
    normalized = series.astype(str).str.strip().str.lower()
    named_labels = {
        "normal": 0,
        "negative": 0,
        "0": 0,
        "abnormal": 1,
        "positive": 1,
        "1": 1,
        "true": 1,
        "false": 0,
    }
    if normalized.isin(named_labels).all():
        return normalized.map(named_labels).astype(int).to_numpy()

    try:
        numeric = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Study labels must be binary 0/1 or NORMAL/ABNORMAL") from error
    if not np.isin(numeric, [0.0, 1.0]).all():
        raise ValueError("Study labels must be binary 0/1")
    return numeric.astype(int)


def _probabilities(values: Sequence[object] | np.ndarray | pd.Series) -> np.ndarray:
    try:
        probabilities = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(
            dtype=float
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Predicted probabilities must be numeric") from error
    if not np.isfinite(probabilities).all():
        raise ValueError("Predicted probabilities contain a non-finite value")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Predicted probabilities must lie between 0 and 1")
    return probabilities


def _labels_and_probabilities(
    y_true: Sequence[object] | np.ndarray | pd.Series,
    y_score: Sequence[object] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    labels = _binary_labels(y_true)
    probabilities = _probabilities(y_score)
    if labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Labels and probabilities must be one-dimensional")
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities must have equal length")
    if len(labels) == 0:
        raise ValueError("At least one study prediction is required")
    if len(np.unique(labels)) != 2:
        raise ValueError("Metrics require at least one NORMAL and one ABNORMAL study")
    return labels, probabilities


def average_precision_score(
    y_true: Sequence[object] | np.ndarray | pd.Series,
    y_score: Sequence[object] | np.ndarray | pd.Series,
) -> float:
    """Return AUPRC / average precision for binary study-level predictions."""

    labels, probabilities = _labels_and_probabilities(y_true, y_score)
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_labels = labels[order]
    sorted_probabilities = probabilities[order]
    true_positives = np.cumsum(sorted_labels)
    false_positives = np.cumsum(1 - sorted_labels)

    # Take one point after each group of tied scores, matching threshold-based
    # precision-recall evaluation rather than depending on sort order in ties.
    last_tie_indices = np.flatnonzero(
        np.r_[sorted_probabilities[:-1] != sorted_probabilities[1:], True]
    )
    tp = true_positives[last_tie_indices]
    fp = false_positives[last_tie_indices]
    precision = tp / (tp + fp)
    recall = tp / int(labels.sum())
    return float(np.sum(precision * np.diff(np.r_[0.0, recall])))


def binary_confusion_counts(
    y_true: Sequence[object] | np.ndarray | pd.Series,
    y_score: Sequence[object] | np.ndarray | pd.Series,
    *,
    threshold: float = 0.5,
) -> dict[str, int | list[list[int]]]:
    """Return TN, FP, FN, TP at a stated probability threshold."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    labels, probabilities = _labels_and_probabilities(y_true, y_score)
    predicted = (probabilities >= threshold).astype(int)
    true_negative = int(np.sum((labels == 0) & (predicted == 0)))
    false_positive = int(np.sum((labels == 0) & (predicted == 1)))
    false_negative = int(np.sum((labels == 1) & (predicted == 0)))
    true_positive = int(np.sum((labels == 1) & (predicted == 1)))
    return {
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "tp": true_positive,
        "matrix": [[true_negative, false_positive], [false_negative, true_positive]],
    }


def compute_study_metrics(
    y_true: Sequence[object] | np.ndarray | pd.Series,
    y_score: Sequence[object] | np.ndarray | pd.Series,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute all required Phase-1 metrics at one operating threshold."""

    labels, probabilities = _labels_and_probabilities(y_true, y_score)
    confusion = binary_confusion_counts(labels, probabilities, threshold=threshold)
    tp, tn = int(confusion["tp"]), int(confusion["tn"])
    fp, fn = int(confusion["fp"]), int(confusion["fn"])
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2.0 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0

    return {
        "n_studies": int(len(labels)),
        "n_normal": int(np.sum(labels == 0)),
        "n_abnormal": int(np.sum(labels == 1)),
        "prevalence_abnormal": float(np.mean(labels)),
        "threshold": float(threshold),
        "auroc": float(binary_roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "recall": float(sensitivity),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "false_negative_rate": float(1.0 - sensitivity),
        "confusion_matrix": confusion,
    }


def _require_prediction_columns(
    predictions: pd.DataFrame,
    *,
    study_id_column: str,
    label_column: str,
    probability_column: str,
) -> None:
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame")
    missing = [
        column
        for column in (study_id_column, label_column, probability_column)
        if column not in predictions.columns
    ]
    if missing:
        raise ValueError("Predictions are missing required column(s): " + ", ".join(missing))


def validate_study_predictions(
    predictions: pd.DataFrame,
    *,
    study_id_column: str = "study_id",
    label_column: str = "label",
    probability_column: str = "probability",
) -> pd.DataFrame:
    """Validate and return a copy of a one-row-per-study prediction frame."""

    _require_prediction_columns(
        predictions,
        study_id_column=study_id_column,
        label_column=label_column,
        probability_column=probability_column,
    )
    if predictions.empty:
        raise ValueError("Predictions are empty")
    if predictions[study_id_column].isna().any():
        raise ValueError("Study predictions contain a missing study_id")
    duplicates = predictions.loc[
        predictions[study_id_column].duplicated(keep=False), study_id_column
    ].astype(str)
    if not duplicates.empty:
        examples = ", ".join(duplicates.drop_duplicates().head(10).tolist())
        raise ValueError(
            "Evaluation requires one row per study; duplicate study_id(s): " + examples
        )
    _labels_and_probabilities(predictions[label_column], predictions[probability_column])
    return predictions.copy()


def aggregate_view_predictions(
    view_predictions: pd.DataFrame,
    *,
    method: str = "mean",
    aggregation: str | None = None,
    study_id_column: str = "study_id",
    label_column: str = "label",
    probability_column: str = "probability",
    metadata_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Aggregate per-view probabilities into a study-level prediction frame.

    ``mean`` and ``max`` are deliberately the only supported Phase-1 choices.
    Any metadata retained in the output must be internally consistent within a
    study (for example, a patient ID or final split).
    """

    if aggregation is not None:
        if method != "mean" and method != aggregation:
            raise ValueError("Specify either method or aggregation, not conflicting values")
        method = aggregation
    method = method.lower()
    if method not in {"mean", "max"}:
        raise ValueError("method must be 'mean' or 'max'")
    _require_prediction_columns(
        view_predictions,
        study_id_column=study_id_column,
        label_column=label_column,
        probability_column=probability_column,
    )
    missing_metadata = [
        column for column in metadata_columns if column not in view_predictions.columns
    ]
    if missing_metadata:
        raise ValueError("Missing requested metadata column(s): " + ", ".join(missing_metadata))
    if view_predictions[study_id_column].isna().any():
        raise ValueError("View predictions contain a missing study_id")
    _probabilities(view_predictions[probability_column])

    grouped = view_predictions.groupby(study_id_column, sort=True, dropna=False)
    inconsistent_labels = grouped[label_column].nunique(dropna=False)
    offenders = inconsistent_labels[inconsistent_labels > 1].index.tolist()
    if offenders:
        raise ValueError(
            "Cannot aggregate inconsistent study labels: "
            + ", ".join(str(item) for item in offenders[:10])
        )
    for column in metadata_columns:
        inconsistent = grouped[column].nunique(dropna=False)
        offenders = inconsistent[inconsistent > 1].index.tolist()
        if offenders:
            raise ValueError(
                f"Cannot aggregate inconsistent {column} values: "
                + ", ".join(str(item) for item in offenders[:10])
            )

    reducer = "mean" if method == "mean" else "max"
    result = grouped.agg(
        **{
            label_column: (label_column, "first"),
            probability_column: (probability_column, reducer),
            "num_views": (probability_column, "size"),
        }
    ).reset_index()
    for column in metadata_columns:
        result[column] = grouped[column].first().to_numpy()
    return result


def evaluate_study_predictions(
    predictions: pd.DataFrame,
    *,
    threshold: float = 0.5,
    study_id_column: str = "study_id",
    label_column: str = "label",
    probability_column: str = "probability",
) -> dict[str, Any]:
    """Evaluate a one-row-per-study frame at the requested threshold."""

    checked = validate_study_predictions(
        predictions,
        study_id_column=study_id_column,
        label_column=label_column,
        probability_column=probability_column,
    )
    return compute_study_metrics(
        checked[label_column], checked[probability_column], threshold=threshold
    )


def _verify_prediction_source(
    predictions: pd.DataFrame,
    *,
    source_split: str,
    expected_split: str,
    split_column: str | None,
) -> None:
    if _normalise_split(source_split) != _normalise_split(expected_split):
        raise ValueError(
            f"This operation requires {expected_split!r} data, not {source_split!r} data."
        )
    if split_column is not None and split_column in predictions.columns:
        observed = {
            _normalise_split(value)
            for value in predictions[split_column].dropna().unique().tolist()
        }
        if observed != {_normalise_split(expected_split)}:
            raise ValueError(
                f"Prediction frame {split_column!r} must contain only {expected_split!r}; "
                f"found {sorted(observed)!r}."
            )


def select_validation_threshold(
    validation_predictions: pd.DataFrame,
    *,
    criterion: str = "youden",
    min_specificity: float | None = None,
    source_split: str = "validation",
    split_column: str | None = "final_split",
    study_id_column: str = "study_id",
    label_column: str = "label",
    probability_column: str = "probability",
) -> dict[str, Any]:
    """Select a deterministic operating threshold using validation data only.

    The default criterion is Youden's J (sensitivity + specificity - 1).  On
    ties the lower threshold is chosen deterministically, which preserves more
    sensitivity among otherwise identical validation objectives.  The returned
    record explicitly identifies validation as the sole selection source.
    """

    _verify_prediction_source(
        validation_predictions,
        source_split=source_split,
        expected_split="validation",
        split_column=split_column,
    )
    checked = validate_study_predictions(
        validation_predictions,
        study_id_column=study_id_column,
        label_column=label_column,
        probability_column=probability_column,
    )
    labels, probabilities = _labels_and_probabilities(
        checked[label_column], checked[probability_column]
    )
    criterion = criterion.lower().replace("-", "_")
    valid_criteria = {"youden", "f1", "balanced_accuracy", "sensitivity_at_specificity"}
    if criterion not in valid_criteria:
        raise ValueError(f"criterion must be one of {sorted(valid_criteria)}")
    if min_specificity is not None and not 0.0 <= min_specificity <= 1.0:
        raise ValueError("min_specificity must be between 0 and 1")
    if criterion == "sensitivity_at_specificity" and min_specificity is None:
        raise ValueError("min_specificity is required for sensitivity_at_specificity")

    candidate_thresholds = np.unique(np.r_[0.0, 1.0, probabilities])
    candidates: list[dict[str, Any]] = []
    for candidate in candidate_thresholds:
        metrics = compute_study_metrics(labels, probabilities, threshold=float(candidate))
        if min_specificity is not None and metrics["specificity"] < min_specificity:
            continue
        if criterion == "youden":
            objective = metrics["sensitivity"] + metrics["specificity"] - 1.0
        elif criterion == "f1":
            objective = metrics["f1"]
        elif criterion == "balanced_accuracy":
            objective = (metrics["sensitivity"] + metrics["specificity"]) / 2.0
        else:
            objective = metrics["sensitivity"]
        candidates.append({"objective": float(objective), "metrics": metrics})
    if not candidates:
        raise ValueError("No threshold satisfies the requested validation constraints")

    # Candidate thresholds were generated in ascending order, so sorting by a
    # negative objective only makes the lowest threshold win an exact tie.
    best = sorted(
        candidates,
        key=lambda candidate: (-candidate["objective"], candidate["metrics"]["threshold"]),
    )[0]
    return {
        "threshold": float(best["metrics"]["threshold"]),
        "criterion": criterion,
        "objective": float(best["objective"]),
        "selection_source": "validation",
        "tie_breaker": "lowest_threshold",
        "validation_metrics": best["metrics"],
    }


def evaluate_test_with_validation_threshold(
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    *,
    threshold_criterion: str = "youden",
    min_specificity: float | None = None,
    validation_source_split: str = "validation",
    test_source_split: str = "test",
    split_column: str | None = "final_split",
    study_id_column: str = "study_id",
    label_column: str = "label",
    probability_column: str = "probability",
    bootstrap_samples: int = 2_000,
    bootstrap_confidence_level: float = 0.95,
    random_state: int | None = 42,
) -> dict[str, Any]:
    """Evaluate test studies at 0.5 and a threshold chosen only on validation."""

    selection = select_validation_threshold(
        validation_predictions,
        criterion=threshold_criterion,
        min_specificity=min_specificity,
        source_split=validation_source_split,
        split_column=split_column,
        study_id_column=study_id_column,
        label_column=label_column,
        probability_column=probability_column,
    )
    _verify_prediction_source(
        test_predictions,
        source_split=test_source_split,
        expected_split="test",
        split_column=split_column,
    )
    checked_test = validate_study_predictions(
        test_predictions,
        study_id_column=study_id_column,
        label_column=label_column,
        probability_column=probability_column,
    )
    at_half = evaluate_study_predictions(
        checked_test,
        threshold=0.5,
        study_id_column=study_id_column,
        label_column=label_column,
        probability_column=probability_column,
    )
    at_selected = evaluate_study_predictions(
        checked_test,
        threshold=selection["threshold"],
        study_id_column=study_id_column,
        label_column=label_column,
        probability_column=probability_column,
    )
    if bootstrap_samples > 0:
        ci = bootstrap_auroc_ci(
            checked_test[label_column],
            checked_test[probability_column],
            n_bootstraps=bootstrap_samples,
            confidence_level=bootstrap_confidence_level,
            random_state=random_state,
        )
        # AUROC and its interval are threshold-independent.  Store the same
        # study-level estimate at both operating points for transparent output.
        at_half["auroc_bootstrap_ci"] = ci
        at_selected["auroc_bootstrap_ci"] = ci

    return {
        "prediction_unit": "study",
        "test_split": "test",
        "threshold_0_5": at_half,
        "validation_selected_threshold": selection,
        "validation_selected": at_selected,
    }


def evaluate_test_with_frozen_validation_threshold(
    test_predictions: pd.DataFrame,
    *,
    frozen_threshold: float,
    frozen_selection_metadata: Mapping[str, Any] | None = None,
    test_source_split: str = "test",
    split_column: str | None = "final_split",
    study_id_column: str = "study_id",
    label_column: str = "label",
    probability_column: str = "probability",
    bootstrap_samples: int = 2_000,
    bootstrap_confidence_level: float = 0.95,
    random_state: int | None = 42,
) -> dict[str, Any]:
    """Evaluate sealed test predictions at a threshold frozen on validation.

    Unlike :func:`evaluate_test_with_validation_threshold`, this function does
    not inspect validation predictions or make a new selection decision.  It
    is therefore the appropriate final-evaluation path after a training run
    has persisted its validation threshold record.
    """

    if not 0.0 <= float(frozen_threshold) <= 1.0:
        raise ValueError("frozen_threshold must be between 0 and 1")
    _verify_prediction_source(
        test_predictions,
        source_split=test_source_split,
        expected_split="test",
        split_column=split_column,
    )
    checked_test = validate_study_predictions(
        test_predictions,
        study_id_column=study_id_column,
        label_column=label_column,
        probability_column=probability_column,
    )
    at_half = evaluate_study_predictions(
        checked_test,
        threshold=0.5,
        study_id_column=study_id_column,
        label_column=label_column,
        probability_column=probability_column,
    )
    at_selected = evaluate_study_predictions(
        checked_test,
        threshold=float(frozen_threshold),
        study_id_column=study_id_column,
        label_column=label_column,
        probability_column=probability_column,
    )
    if bootstrap_samples > 0:
        ci = bootstrap_auroc_ci(
            checked_test[label_column],
            checked_test[probability_column],
            n_bootstraps=bootstrap_samples,
            confidence_level=bootstrap_confidence_level,
            random_state=random_state,
        )
        at_half["auroc_bootstrap_ci"] = ci
        at_selected["auroc_bootstrap_ci"] = ci

    selection: dict[str, Any] = {
        "threshold": float(frozen_threshold),
        "selection_source": "validation",
        "selection_mode": "frozen_validation_record",
    }
    if frozen_selection_metadata is not None:
        selection["frozen_selection_metadata"] = dict(frozen_selection_metadata)
    return {
        "prediction_unit": "study",
        "test_split": "test",
        "threshold_0_5": at_half,
        "validation_selected_threshold": selection,
        "validation_selected": at_selected,
    }


def _metrics_for_operating_point(
    evaluation: Mapping[str, Any], operating_point: str
) -> Mapping[str, Any]:
    aliases = {
        "validation_selected_threshold": "validation_selected",
        "selected": "validation_selected",
        "0.5": "threshold_0_5",
        "0_5": "threshold_0_5",
    }
    key = aliases.get(operating_point, operating_point)
    if key in evaluation and isinstance(evaluation[key], Mapping):
        return evaluation[key]
    # A single operating-point metrics dictionary is also accepted.
    if "auroc" in evaluation and "auprc" in evaluation:
        return evaluation
    raise ValueError(f"Evaluation does not contain operating point {operating_point!r}")


def build_model_comparison_table(
    evaluations: Mapping[str, Mapping[str, Any]],
    *,
    operating_point: str = "validation_selected",
) -> pd.DataFrame:
    """Build the required study-level comparison table for one operating point."""

    rows: list[dict[str, Any]] = []
    for model_name, evaluation in evaluations.items():
        metrics = _metrics_for_operating_point(evaluation, operating_point)
        confusion = metrics.get("confusion_matrix", {})
        rows.append(
            {
                "Model": str(model_name),
                "Threshold": metrics.get("threshold"),
                "AUROC": metrics.get("auroc"),
                "AUPRC": metrics.get("auprc"),
                "Recall": metrics.get("recall", metrics.get("sensitivity")),
                "Sensitivity": metrics.get("sensitivity", metrics.get("recall")),
                "Specificity": metrics.get("specificity"),
                "F1": metrics.get("f1"),
                "FN Rate": metrics.get("false_negative_rate"),
                "Studies": metrics.get("n_studies"),
                "TN": confusion.get("tn"),
                "FP": confusion.get("fp"),
                "FN": confusion.get("fn"),
                "TP": confusion.get("tp"),
            }
        )
    if not rows:
        raise ValueError("At least one model evaluation is required")
    return pd.DataFrame(rows)


def _safe_filename(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return name.strip("._") or "model"


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _operating_points_frame(evaluation: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in ("threshold_0_5", "validation_selected"):
        metrics = evaluation.get(name)
        if not isinstance(metrics, Mapping):
            continue
        row = {"operating_point": name}
        for key, value in metrics.items():
            row[key] = json.dumps(value, default=_json_default) if isinstance(value, Mapping) else value
        rows.append(row)
    return pd.DataFrame(rows)


def persist_evaluation_results(
    model_name: str,
    evaluation: Mapping[str, Any],
    output_dir: str | Path,
    *,
    comparison_table: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Persist a model's complete results and optional comparison table.

    JSON preserves the nested confusion matrix and CI; CSV is convenient for
    the report and spreadsheet review.  Existing files with the same model
    name are intentionally replaced by this explicit evaluation run.
    """

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stem = _safe_filename(model_name)
    evaluation_json = target / f"{stem}_evaluation.json"
    operating_points_csv = target / f"{stem}_operating_points.csv"
    with evaluation_json.open("w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, indent=2, default=_json_default)
        handle.write("\n")
    _operating_points_frame(evaluation).to_csv(operating_points_csv, index=False)

    paths = {
        "evaluation_json": str(evaluation_json),
        "operating_points_csv": str(operating_points_csv),
    }
    if comparison_table is not None:
        comparison_path = target / "model_comparison.csv"
        comparison_table.to_csv(comparison_path, index=False)
        paths["model_comparison_csv"] = str(comparison_path)
    return paths


# Convenient aliases for notebooks and scripts.
evaluate_predictions = evaluate_study_predictions
compare_models = build_model_comparison_table
