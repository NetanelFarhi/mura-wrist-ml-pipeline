"""Descriptive, study-level error-analysis indices for Phase 1.

The outputs identify observable labels, model probabilities, thresholds, and
view counts.  They intentionally do not infer a clinical cause for an error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .model_eval import _binary_labels, _probabilities, validate_study_predictions


def annotate_prediction_outcomes(
    predictions: pd.DataFrame,
    *,
    threshold: float,
    study_id_column: str = "study_id",
    label_column: str = "label",
    probability_column: str = "probability",
    num_views_column: str = "num_views",
) -> pd.DataFrame:
    """Add thresholded outcomes and confidence fields to study predictions."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    checked = validate_study_predictions(
        predictions,
        study_id_column=study_id_column,
        label_column=label_column,
        probability_column=probability_column,
    )
    if num_views_column not in checked.columns:
        raise ValueError(
            f"Error analysis requires {num_views_column!r} for single/multi-view analysis"
        )
    try:
        num_views = pd.to_numeric(checked[num_views_column], errors="raise").to_numpy(
            dtype=float
        )
    except (TypeError, ValueError) as error:
        raise ValueError("num_views must be numeric") from error
    if not np.isfinite(num_views).all() or (num_views < 1).any() or (num_views % 1 != 0).any():
        raise ValueError("num_views must be positive whole numbers")

    labels = _binary_labels(checked[label_column])
    probabilities = _probabilities(checked[probability_column])
    predicted = (probabilities >= threshold).astype(int)
    outcome = np.select(
        [
            (labels == 0) & (predicted == 0),
            (labels == 0) & (predicted == 1),
            (labels == 1) & (predicted == 0),
            (labels == 1) & (predicted == 1),
        ],
        ["true_negative", "false_positive", "false_negative", "true_positive"],
        default="unclassified",
    )
    annotated = checked.copy()
    annotated["true_label"] = labels
    annotated["predicted_label"] = predicted
    annotated["outcome"] = outcome
    annotated["prediction_confidence"] = np.where(predicted == 1, probabilities, 1.0 - probabilities)
    annotated["distance_to_threshold"] = np.abs(probabilities - threshold)
    annotated[num_views_column] = num_views.astype(int)
    return annotated


def _representative(
    frame: pd.DataFrame,
    *,
    max_examples: int,
    study_id_column: str,
    ascending_confidence: bool = False,
) -> pd.DataFrame:
    """Return a deterministic, compact index rather than image content."""

    if frame.empty:
        return frame.copy()
    return frame.sort_values(
        ["prediction_confidence", "distance_to_threshold", study_id_column],
        ascending=[ascending_confidence, True, True],
        kind="mergesort",
    ).head(max_examples).copy()


def build_error_analysis_index(
    predictions: pd.DataFrame,
    *,
    threshold: float,
    high_confidence: float = 0.80,
    low_confidence_max: float = 0.60,
    max_examples_per_category: int = 25,
    study_id_column: str = "study_id",
    label_column: str = "label",
    probability_column: str = "probability",
    num_views_column: str = "num_views",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Create representative indices for required Phase-1 error categories.

    High-confidence false negatives are ABNORMAL studies predicted NORMAL with
    predicted-NORMAL confidence at least ``high_confidence``; high-confidence
    false positives use the analogous predicted-ABNORMAL confidence.  "Low
    confidence" denotes probabilities near 0.5, not a clinical uncertainty
    estimate.
    """

    if not 0.5 <= high_confidence <= 1.0:
        raise ValueError("high_confidence must be between 0.5 and 1")
    if not 0.5 <= low_confidence_max <= 1.0:
        raise ValueError("low_confidence_max must be between 0.5 and 1")
    if max_examples_per_category <= 0:
        raise ValueError("max_examples_per_category must be positive")

    annotated = annotate_prediction_outcomes(
        predictions,
        threshold=threshold,
        study_id_column=study_id_column,
        label_column=label_column,
        probability_column=probability_column,
        num_views_column=num_views_column,
    )
    false_negatives = annotated.loc[annotated["outcome"] == "false_negative"]
    false_positives = annotated.loc[annotated["outcome"] == "false_positive"]
    is_low_confidence = annotated["prediction_confidence"] <= low_confidence_max
    categories = {
        "false_negatives": _representative(
            false_negatives,
            max_examples=max_examples_per_category,
            study_id_column=study_id_column,
        ),
        "false_positives": _representative(
            false_positives,
            max_examples=max_examples_per_category,
            study_id_column=study_id_column,
        ),
        "high_confidence_false_negatives": _representative(
            false_negatives.loc[
                false_negatives["prediction_confidence"] >= high_confidence
            ],
            max_examples=max_examples_per_category,
            study_id_column=study_id_column,
        ),
        "high_confidence_false_positives": _representative(
            false_positives.loc[
                false_positives["prediction_confidence"] >= high_confidence
            ],
            max_examples=max_examples_per_category,
            study_id_column=study_id_column,
        ),
        "low_confidence_predictions": _representative(
            annotated.loc[is_low_confidence],
            max_examples=max_examples_per_category,
            study_id_column=study_id_column,
            ascending_confidence=True,
        ),
        "single_view_errors": _representative(
            annotated.loc[
                (annotated[num_views_column] == 1)
                & annotated["outcome"].isin(["false_negative", "false_positive"])
            ],
            max_examples=max_examples_per_category,
            study_id_column=study_id_column,
        ),
        "multi_view_errors": _representative(
            annotated.loc[
                (annotated[num_views_column] > 1)
                & annotated["outcome"].isin(["false_negative", "false_positive"])
            ],
            max_examples=max_examples_per_category,
            study_id_column=study_id_column,
        ),
        "single_view_studies": _representative(
            annotated.loc[annotated[num_views_column] == 1],
            max_examples=max_examples_per_category,
            study_id_column=study_id_column,
        ),
        "multi_view_studies": _representative(
            annotated.loc[annotated[num_views_column] > 1],
            max_examples=max_examples_per_category,
            study_id_column=study_id_column,
        ),
    }
    return annotated, categories


def summarize_error_analysis(
    annotated_predictions: pd.DataFrame,
    *,
    num_views_column: str = "num_views",
) -> pd.DataFrame:
    """Summarize observable error counts by outcome and view-count grouping."""

    if num_views_column not in annotated_predictions.columns:
        raise ValueError(f"Missing {num_views_column!r} in annotated predictions")
    if "outcome" not in annotated_predictions.columns:
        raise ValueError("annotated_predictions must contain an outcome column")
    view_group = np.where(
        annotated_predictions[num_views_column].to_numpy() == 1,
        "single_view",
        "multi_view",
    )
    summary_frame = annotated_predictions.assign(view_group=view_group)
    counts = (
        summary_frame.groupby(["view_group", "outcome"], dropna=False)
        .size()
        .rename("n_studies")
        .reset_index()
    )
    total = pd.DataFrame(
        {
            "view_group": ["all_studies"],
            "outcome": ["all"],
            "n_studies": [int(len(annotated_predictions))],
        }
    )
    return pd.concat([counts, total], ignore_index=True)


def _markdown_summary(
    summary: pd.DataFrame,
    categories: dict[str, pd.DataFrame],
    *,
    threshold: float,
    high_confidence: float,
    low_confidence_max: float,
) -> str:
    """Render a dependency-free descriptive report; no causal interpretations."""

    lines = [
        "# Phase 1 error-analysis index",
        "",
        f"Study-level operating threshold: `{threshold:.4f}`.",
        "",
        "This report is descriptive: it indexes observed labels, model probabilities, "
        "and view counts. It does not establish a clinical cause for any false negative "
        "or false positive.",
        "",
        "## Observable outcome counts",
        "",
        "| View grouping | Outcome | Studies |",
        "|---|---|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.view_group} | {row.outcome} | {int(row.n_studies)} |")
    lines.extend(
        [
            "",
            "## Representative index categories",
            "",
            "| Category | Indexed studies | Definition |",
            "|---|---:|---|",
        ]
    )
    definitions = {
        "false_negatives": "ABNORMAL study predicted NORMAL at the operating threshold.",
        "false_positives": "NORMAL study predicted ABNORMAL at the operating threshold.",
        "high_confidence_false_negatives": (
            f"False negative with predicted-NORMAL confidence at least {high_confidence:.2f}."
        ),
        "high_confidence_false_positives": (
            f"False positive with predicted-ABNORMAL confidence at least {high_confidence:.2f}."
        ),
        "low_confidence_predictions": (
            f"Probability confidence no greater than {low_confidence_max:.2f}; this is not a clinical uncertainty estimate."
        ),
        "single_view_errors": "False negatives or false positives from one-view studies.",
        "multi_view_errors": "False negatives or false positives from multi-view studies.",
        "single_view_studies": "Representative one-view studies, regardless of outcome.",
        "multi_view_studies": "Representative multi-view studies, regardless of outcome.",
    }
    for category, frame in categories.items():
        lines.append(f"| {category} | {len(frame)} | {definitions[category]} |")
    lines.append("")
    return "\n".join(lines)


def write_error_analysis_reports(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    *,
    threshold: float,
    high_confidence: float = 0.80,
    low_confidence_max: float = 0.60,
    max_examples_per_category: int = 25,
    study_id_column: str = "study_id",
    label_column: str = "label",
    probability_column: str = "probability",
    num_views_column: str = "num_views",
    filename_prefix: str = "",
) -> dict[str, str]:
    """Write CSV/Markdown error indices and return their paths."""

    annotated, categories = build_error_analysis_index(
        predictions,
        threshold=threshold,
        high_confidence=high_confidence,
        low_confidence_max=low_confidence_max,
        max_examples_per_category=max_examples_per_category,
        study_id_column=study_id_column,
        label_column=label_column,
        probability_column=probability_column,
        num_views_column=num_views_column,
    )
    summary = summarize_error_analysis(annotated, num_views_column=num_views_column)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    prefix = f"{filename_prefix}_" if filename_prefix else ""

    annotations_path = target / f"{prefix}study_error_annotations.csv"
    summary_path = target / f"{prefix}error_analysis_summary.csv"
    index_path = target / f"{prefix}error_analysis_index.csv"
    markdown_path = target / f"{prefix}error_analysis.md"
    annotated.to_csv(annotations_path, index=False)
    summary.to_csv(summary_path, index=False)

    category_frames: list[pd.DataFrame] = []
    for category, frame in categories.items():
        indexed = frame.copy()
        indexed.insert(0, "category", category)
        indexed.insert(1, "rank_within_category", range(1, len(indexed) + 1))
        category_frames.append(indexed)
    if category_frames:
        pd.concat(category_frames, ignore_index=True).to_csv(index_path, index=False)
    else:
        pd.DataFrame(columns=["category", "rank_within_category"]).to_csv(index_path, index=False)
    markdown_path.write_text(
        _markdown_summary(
            summary,
            categories,
            threshold=threshold,
            high_confidence=high_confidence,
            low_confidence_max=low_confidence_max,
        ),
        encoding="utf-8",
    )
    return {
        "annotations_csv": str(annotations_path),
        "summary_csv": str(summary_path),
        "index_csv": str(index_path),
        "markdown": str(markdown_path),
    }
