#!/usr/bin/env python
"""Build safe, aggregate-only Checkpoint 1 results for GitHub.

This utility deliberately reads *only* internal-validation prediction files
from a locally completed Model A / Model B run.  It never opens an official
MURA ``valid``/test prediction file and it writes no radiographs, study IDs,
patient IDs, local paths, checkpoints, or per-study probabilities.

Example
-------
python scripts/build_checkpoint1_results.py \
  --model-a-dir C:/local/model_a/out \
  --model-b-dir C:/local/model_b/out \
  --output-dir docs/results/checkpoint1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


REQUIRED_COLUMNS = {
    "patient_id",
    "study_id",
    "label",
    "num_views",
    "abnormal_probability",
}


@dataclass(frozen=True)
class ModelPredictions:
    """Validation predictions retained only in memory while reporting."""

    name: str
    rows: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a-dir", type=Path, required=True)
    parser.add_argument("--model-b-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--shortcut-audit-dir",
        type=Path,
        help="Optional local diagnostic-audit directory; only aggregate perturbation fields are published.",
    )
    parser.add_argument("--sensitivity-target", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _assert_internal_validation_path(path: Path) -> None:
    """Fail closed rather than accidentally creating a report from test data."""

    normalized = str(path).replace("\\", "/").lower()
    banned_tokens = ("test_predictions", "official_valid", "official-valid", "/valid/")
    if any(token in normalized for token in banned_tokens):
        raise ValueError(f"Refusing a possible official-valid/test input: {path}")
    if path.name not in {
        "validation_predictions.csv",
        "validation_predictions_mean.csv",
        "validation_predictions_max.csv",
    }:
        raise ValueError(f"Expected a validation prediction artifact, got: {path.name}")


def load_validation_predictions(name: str, path: Path) -> ModelPredictions:
    _assert_internal_validation_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(rows.columns)
    if missing:
        raise ValueError(f"{path.name} is missing: {sorted(missing)}")
    if rows["study_id"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate study predictions.")
    if not rows["label"].isin([0, 1]).all():
        raise ValueError(f"{path.name} labels must be binary.")
    if set(rows["label"].unique()) != {0, 1}:
        raise ValueError(f"{path.name} must contain both NORMAL and ABNORMAL studies.")
    if not rows["num_views"].ge(1).all():
        raise ValueError(f"{path.name} contains a study with no real views.")
    if not rows["abnormal_probability"].between(0, 1).all():
        raise ValueError(f"{path.name} has invalid probabilities.")
    return ModelPredictions(name=name, rows=rows.copy())


def _load_validation_selection(model_dir: Path, *, model_name: str) -> dict:
    """Require selection provenance compatible with the current protocol.

    Model A must select ``C`` in patient-grouped CV inside internal training.
    Model B may use internal validation for checkpoint selection, but neither
    model may use official-valid/test data. This cannot prove an arbitrarily
    renamed CSV's provenance; it does fail closed when a record contradicts
    the documented protocol.
    """

    selection_path = model_dir / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"Missing selection record for {model_name}: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("test_used_for_selection") is not False:
        raise ValueError(f"{model_name} selection record does not rule out test use.")
    if model_name == "Model A":
        if selection.get("c_selection_data") != "internal_train_patient_grouped_cv_only":
            raise ValueError("Model A C selection is not recorded as internal-train grouped CV.")
        if selection.get("internal_validation_labels_used_for_c_selection") is not False:
            raise ValueError("Model A selection record does not rule out validation-label use for C.")
        if selection.get("threshold_selection_data") != "internal_validation_only":
            raise ValueError("Model A threshold is not recorded as internal-validation only.")
    elif model_name == "Model B" and selection.get("selection_data") != "validation only":
        raise ValueError("Model B selection record is not validation-only.")
    return selection


def _assert_model_b_run_matches_validation(model_dir: Path, *, expected_studies: int) -> None:
    """Check the saved Model B run contract before publishing aggregate results."""

    run_config_path = model_dir / "run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"Missing Model B run configuration: {run_config_path}")
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    if int(run_config.get("validation_studies", -1)) != expected_studies:
        raise ValueError("Model B run configuration does not match the validation prediction cohort.")
    arguments = run_config.get("arguments", {})
    if bool(arguments.get("write_test_predictions", False)):
        raise ValueError("Refusing a Model B run configured to write test predictions.")


def assert_paired_cohort(predictions: Iterable[ModelPredictions]) -> None:
    items = list(predictions)
    reference = items[0].rows.set_index("study_id").sort_index()
    for item in items[1:]:
        candidate = item.rows.set_index("study_id").sort_index()
        if not reference.index.equals(candidate.index):
            raise ValueError(f"{item.name} is not evaluated on the same validation studies.")
        for column in ("patient_id", "label", "num_views"):
            if not reference[column].equals(candidate[column]):
                raise ValueError(f"{item.name} disagrees with Model A on {column!r}.")


def classification_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = (scores >= threshold).astype(int)
    true_positive = int(np.sum((labels == 1) & (predicted == 1)))
    true_negative = int(np.sum((labels == 0) & (predicted == 0)))
    false_positive = int(np.sum((labels == 0) & (predicted == 1)))
    false_negative = int(np.sum((labels == 1) & (predicted == 0)))
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    return {
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "false_positive": false_positive,
        "false_negative": false_negative,
        "flagged_abnormal": int(predicted.sum()),
        "flagged_percent": float(100 * predicted.mean()),
    }


def highest_threshold_for_sensitivity(
    labels: np.ndarray, scores: np.ndarray, target: float
) -> dict[str, float | int]:
    if not 0 < target <= 1:
        raise ValueError("--sensitivity-target must be in (0, 1].")
    candidates = np.unique(scores)[::-1]
    eligible = [
        classification_at_threshold(labels, scores, float(threshold))
        for threshold in candidates
        if classification_at_threshold(labels, scores, float(threshold))["sensitivity"] >= target
    ]
    if not eligible:
        raise RuntimeError(f"No validation threshold reached sensitivity >= {target:.0%}.")
    # Candidates are descending, so the first one is the most selective policy
    # that still meets the pre-specified sensitivity constraint.
    return eligible[0]


def patient_bootstrap_intervals(
    rows: pd.DataFrame, *, threshold: float, samples: int, seed: int
) -> dict[str, tuple[float, float]] | None:
    """Patient-cluster bootstrap intervals for ranking and a frozen policy."""

    if samples < 1:
        return None
    patient_codes, _ = pd.factorize(rows["patient_id"].astype(str), sort=False)
    n_patients = int(patient_codes.max()) + 1
    if n_patients < 2:
        return None
    labels = rows["label"].to_numpy(dtype=int)
    scores = rows["abnormal_probability"].to_numpy(dtype=float)
    predicted = scores >= threshold
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"auroc": [], "auprc": [], "sensitivity": [], "specificity": []}
    for _ in range(samples):
        drawn = rng.integers(0, n_patients, size=n_patients)
        weights_by_patient = np.bincount(drawn, minlength=n_patients)
        sample_weight = weights_by_patient[patient_codes]
        if not np.any(sample_weight[labels == 0]) or not np.any(sample_weight[labels == 1]):
            continue
        true_positive = sample_weight[(labels == 1) & predicted].sum()
        false_negative = sample_weight[(labels == 1) & ~predicted].sum()
        true_negative = sample_weight[(labels == 0) & ~predicted].sum()
        false_positive = sample_weight[(labels == 0) & predicted].sum()
        values["auroc"].append(float(roc_auc_score(labels, scores, sample_weight=sample_weight)))
        values["auprc"].append(
            float(average_precision_score(labels, scores, sample_weight=sample_weight))
        )
        values["sensitivity"].append(float(true_positive / (true_positive + false_negative)))
        values["specificity"].append(float(true_negative / (true_negative + false_positive)))
    if not values["auroc"]:
        return None
    return {
        metric: tuple(float(value) for value in np.quantile(metric_values, [0.025, 0.975]))
        for metric, metric_values in values.items()
    }


def summarize_model(
    prediction: ModelPredictions, *, target: float, bootstrap_samples: int, seed: int
) -> dict[str, float | int | str]:
    labels = prediction.rows["label"].to_numpy(dtype=int)
    scores = prediction.rows["abnormal_probability"].to_numpy(dtype=float)
    target_policy = highest_threshold_for_sensitivity(labels, scores, target)
    default_policy = classification_at_threshold(labels, scores, 0.5)
    intervals = patient_bootstrap_intervals(
        prediction.rows,
        threshold=float(target_policy["threshold"]),
        samples=bootstrap_samples,
        seed=seed,
    )
    result: dict[str, float | int | str] = {
        "model": prediction.name,
        "evaluation_population": "internal patient-level validation from official MURA train",
        "studies": int(len(prediction.rows)),
        "patients": int(prediction.rows["patient_id"].nunique()),
        "normal_studies": int((labels == 0).sum()),
        "abnormal_studies": int((labels == 1).sum()),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "auroc_ci_lower": float(intervals["auroc"][0]) if intervals else np.nan,
        "auroc_ci_upper": float(intervals["auroc"][1]) if intervals else np.nan,
        "auprc_ci_lower": float(intervals["auprc"][0]) if intervals else np.nan,
        "auprc_ci_upper": float(intervals["auprc"][1]) if intervals else np.nan,
        "sensitivity_target": float(target),
        "threshold_at_target": target_policy["threshold"],
        "sensitivity_at_target": target_policy["sensitivity"],
        "specificity_at_target": target_policy["specificity"],
        "sensitivity_ci_lower": float(intervals["sensitivity"][0]) if intervals else np.nan,
        "sensitivity_ci_upper": float(intervals["sensitivity"][1]) if intervals else np.nan,
        "specificity_ci_lower": float(intervals["specificity"][0]) if intervals else np.nan,
        "specificity_ci_upper": float(intervals["specificity"][1]) if intervals else np.nan,
        "precision_at_target": target_policy["precision"],
        "f1_at_target": target_policy["f1"],
        "false_positive_at_target": target_policy["false_positive"],
        "false_negative_at_target": target_policy["false_negative"],
        "flagged_percent_at_target": target_policy["flagged_percent"],
        "sensitivity_at_0_5": default_policy["sensitivity"],
        "specificity_at_0_5": default_policy["specificity"],
        "f1_at_0_5": default_policy["f1"],
    }
    return result


def write_figures(
    predictions: list[ModelPredictions], summaries: pd.DataFrame, output_dir: Path, *, target: float
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#1d4ed8", "#059669", "#b45309")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for item, color in zip(predictions, colors):
        labels = item.rows["label"].to_numpy(dtype=int)
        scores = item.rows["abnormal_probability"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        axes[0].plot(fpr, tpr, color=color, lw=2, label=f"{item.name} (AUROC {roc_auc_score(labels, scores):.3f})")
        axes[1].plot(recall, precision, color=color, lw=2, label=f"{item.name} (AUPRC {average_precision_score(labels, scores):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1)
    axes[0].set(xlabel="False-positive rate", ylabel="Sensitivity", title="Validation ROC")
    axes[1].set(xlabel="Sensitivity", ylabel="Precision", title="Validation precision-recall")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].legend(fontsize=8, loc="lower left")

    x = np.arange(len(summaries))
    width = 0.28
    axes[2].bar(x - width, summaries["sensitivity_at_target"], width, label="Sensitivity", color="#1d4ed8")
    axes[2].bar(x, summaries["specificity_at_target"], width, label="Specificity", color="#059669")
    axes[2].bar(x + width, summaries["flagged_percent_at_target"] / 100, width, label="Flagged for review", color="#b45309")
    axes[2].axhline(target, color="#475569", ls="--", lw=1, label=f"{target:.0%} target")
    policy_labels = {
        "Frozen ResNet-18 + Logistic Regression": "Model A\nbaseline",
        "Fine-tuned ResNet-18 mean pooling": "Model B mean\n(primary)",
        "Fine-tuned ResNet-18 max pooling (secondary diagnostic)": "Model B max\n(secondary)",
    }
    axes[2].set(
        xticks=x,
        xticklabels=[policy_labels.get(str(name), str(name)) for name in summaries["model"]],
        ylim=(0, 1.05),
        ylabel="Proportion of validation studies",
        title="High-sensitivity operating policy",
    )
    axes[2].tick_params(axis="x", labelsize=8)
    axes[2].legend(fontsize=8, loc="lower left")
    fig.suptitle("Checkpoint 1 — internal validation only", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "validation_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_training_curve(history_path: Path, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = json.loads(history_path.read_text(encoding="utf-8"))
    entries = history.get("history", [])
    if not entries:
        return
    epochs = [entry["epoch"] for entry in entries]
    loss = [entry["train_loss"] for entry in entries]
    auroc = [entry["validation_monitor"] for entry in entries]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    axes[0].plot(epochs, loss, marker="o", color="#1d4ed8")
    axes[0].set(xlabel="Epoch", ylabel="Study-level training loss", title="Fine-tuning training loss")
    axes[1].plot(epochs, auroc, marker="o", color="#059669")
    axes[1].axvline(history["best_epoch"], color="#475569", ls="--", lw=1, label=f"Best epoch {history['best_epoch']}")
    axes[1].set(xlabel="Epoch", ylabel="Validation AUROC", title="Fine-tuning validation AUROC")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "finetuning_history.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_model_a_grouped_cv_summary(selection: dict, output_dir: Path) -> None:
    """Publish the safe, aggregate C-selection table used by the baseline.

    The table intentionally contains no study, patient, image, or embedding
    data.  It lets the notebook snapshot show that regularization was selected
    inside internal training with patient-grouped cross-validation.
    """

    c_selection = selection.get("c_selection", {})
    rows = pd.DataFrame(c_selection.get("candidate_results", []))
    required = {"C", "train_group_cv_auroc", "train_group_cv_auprc"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Model A grouped-CV record is incomplete: {sorted(missing)}")
    selected_c = float(c_selection.get("selected_c"))
    rows = rows.loc[:, ["C", "train_group_cv_auroc", "train_group_cv_auprc"]].copy()
    rows["selected"] = np.isclose(rows["C"].astype(float), selected_c)
    rows.sort_values("C", inplace=True)
    rows.to_csv(output_dir / "model_a_grouped_cv_candidates.csv", index=False)


def write_markdown(
    summaries: pd.DataFrame, output_dir: Path, target: float, *, include_shortcut_audit: bool
) -> None:
    display_columns = [
        "model", "auroc", "auprc", "threshold_at_target", "sensitivity_at_target",
        "specificity_at_target", "precision_at_target", "f1_at_target",
        "false_negative_at_target", "false_positive_at_target", "flagged_percent_at_target",
    ]
    table = summaries[display_columns].copy()
    for column in table.columns:
        if column not in {"model", "false_negative_at_target", "false_positive_at_target"}:
            table[column] = table[column].map(lambda value: f"{float(value):.3f}")
    markdown_table = table.to_markdown(index=False)
    interval_table = summaries[
        [
            "model",
            "auroc_ci_lower",
            "auroc_ci_upper",
            "auprc_ci_lower",
            "auprc_ci_upper",
            "sensitivity_ci_lower",
            "sensitivity_ci_upper",
            "specificity_ci_lower",
            "specificity_ci_upper",
        ]
    ].copy()
    interval_table["AUROC 95% CI"] = interval_table.apply(
        lambda row: f"{row['auroc_ci_lower']:.3f}–{row['auroc_ci_upper']:.3f}", axis=1
    )
    interval_table["AUPRC 95% CI"] = interval_table.apply(
        lambda row: f"{row['auprc_ci_lower']:.3f}–{row['auprc_ci_upper']:.3f}", axis=1
    )
    interval_table["Sensitivity 95% CI"] = interval_table.apply(
        lambda row: f"{row['sensitivity_ci_lower']:.3f}–{row['sensitivity_ci_upper']:.3f}", axis=1
    )
    interval_table["Specificity 95% CI"] = interval_table.apply(
        lambda row: f"{row['specificity_ci_lower']:.3f}–{row['specificity_ci_upper']:.3f}", axis=1
    )
    interval_markdown = interval_table[
        ["model", "AUROC 95% CI", "AUPRC 95% CI", "Sensitivity 95% CI", "Specificity 95% CI"]
    ].to_markdown(index=False)
    shortcut_section = """
## Diagnostic robustness check

The separate [shortcut and geometry diagnostic](shortcut_audit.md) records
aggregate perturbation results only. It is a model-diagnostic warning check,
not a second performance evaluation or localization result.
""" if include_shortcut_audit else ""

    content = f"""# Checkpoint 1 — internal validation results

This page is generated from local, study-level predictions on the **internal
patient-level validation split drawn from official MURA train**. It contains
aggregate metrics only: no radiographs, identifiers, per-study predictions,
checkpoints, or local paths are committed.

## Evaluation cohort

- {int(summaries.iloc[0]['studies']):,} studies from {int(summaries.iloc[0]['patients']):,} patients
- {int(summaries.iloc[0]['normal_studies']):,} NORMAL and {int(summaries.iloc[0]['abnormal_studies']):,} ABNORMAL studies
- The official MURA `valid` split was **not read** for this Checkpoint 1 run.

## Model comparison

The threshold in this table is the highest internal-validation threshold that
achieves sensitivity ≥ {target:.0%}; it is an operating-policy analysis on the
same development split, not a final clinical claim.

The primary comparison is the required frozen-embedding Logistic Regression
baseline versus the existing fine-tuned ResNet-18 under the revised protocol's
primary mean study-pooling policy. Max pooling is included only as a
transparent secondary diagnostic. The saved Model B weights were not retrained
for this reporting update.

{markdown_table}

## Patient-cluster bootstrap uncertainty

The following 95% intervals resample patients rather than individual views.
Sensitivity and specificity treat the displayed sensitivity-constrained
threshold as fixed within each resample.

{interval_markdown}

![Validation model comparison](validation_model_comparison.png)

## Fine-tuning training history

![Fine-tuning history](finetuning_history.png)
{shortcut_section}

## Interpretation limits

- These are **development/validation** results, not results from an untouched
  final test set.
- MURA `ABNORMAL` does not mean acute fracture, and no model output is a
  diagnosis or localization claim.
- The high-sensitivity policy intentionally raises the proportion of studies
  flagged for human review.
- Patient-cluster bootstrap intervals quantify sampling uncertainty but do not
  turn this split into a final test set.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def write_shortcut_audit_summary(audit_dir: Path, output_dir: Path) -> None:
    """Publish only safe diagnostic aggregates, never audit predictions or identifiers."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_path = audit_dir / "audit_report.json"
    robustness_path = audit_dir / "robustness_summary.csv"
    if not report_path.is_file() or not robustness_path.is_file():
        raise FileNotFoundError("Shortcut audit needs audit_report.json and robustness_summary.csv.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    robustness = pd.read_csv(robustness_path)
    required = {
        "variant",
        "mean_absolute_probability_change",
        "decision_flips_at_frozen_threshold",
        "decision_flip_percent_at_frozen_threshold",
    }
    missing = required - set(robustness.columns)
    if missing:
        raise ValueError(f"Shortcut-audit summary missing columns: {sorted(missing)}")
    safe = robustness.loc[
        robustness["variant"].ne("canonical"),
        [
            "variant",
            "mean_absolute_probability_change",
            "decision_flips_at_frozen_threshold",
            "decision_flip_percent_at_frozen_threshold",
        ],
    ].copy()
    display_names = {
        "padding_midgray": "Padding → gray",
        "outer_border_black_5pct": "5% border mask",
        "direct_resize": "Direct resize",
    }
    safe["display_variant"] = safe["variant"].map(display_names).fillna(safe["variant"])
    safe.to_csv(output_dir / "shortcut_audit_summary.csv", index=False)

    geometry = report.get("geometry_only_grouped_cv", {})
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    axes[0].bar(safe["display_variant"], safe["mean_absolute_probability_change"], color="#2563eb")
    axes[0].set(title="Prediction change under perturbation", ylabel="Mean |Δ probability|")
    axes[1].bar(
        safe["display_variant"], safe["decision_flip_percent_at_frozen_threshold"], color="#b45309"
    )
    axes[1].set(title="Frozen-threshold decision changes", ylabel="Percent of studies")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20, labelsize=8)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout(pad=1.2)
    figure.subplots_adjust(top=0.88)
    figure.savefig(output_dir / "shortcut_audit.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    content = f"""# Shortcut and geometry diagnostic

This diagnostic uses explicit internal-train/internal-validation manifests;
the official MURA `valid` split was rejected and not read. It is not a new
model-performance evaluation and does not show individual radiographs or IDs.

## What was tested

- A geometry-only, patient-grouped cross-validation warning check using view
  count, aspect ratio, and square-padding summaries.
- Controlled changes to padding pixels, an outer-border stress test, and a
  direct-resize stress test on the fixed development model.

The geometry-only warning model produced AUROC
{float(geometry.get('oof_auroc', np.nan)):.3f} and AUPRC
{float(geometry.get('oof_auprc', np.nan)):.3f}. This shows that image
geometry carries label-associated information in this development population;
it **does not** prove that the ResNet uses a shortcut or that it reasons from
anatomy.

| Perturbation | Mean \\|Δ probability\\| | Decision flips at frozen threshold |
|---|---:|---:|
"""
    for _, row in safe.iterrows():
        content += (
            f"| {row['variant']} | {row['mean_absolute_probability_change']:.3f} | "
            f"{int(row['decision_flips_at_frozen_threshold'])} "
            f"({row['decision_flip_percent_at_frozen_threshold']:.2f}%) |\n"
        )
    content += """

![Aggregate shortcut-audit diagnostics](shortcut_audit.png)

The padding intervention is the most isolated frame check. Border masking and
direct resize are stress tests because they can alter image content or geometry.
These results justify preserving aspect-ratio-aware preprocessing and a future
manual/Grad-CAM diagnostic review; they are not lesion-localization evidence.
"""
    (output_dir / "shortcut_audit.md").write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive.")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_a_selection = _load_validation_selection(args.model_a_dir, model_name="Model A")
    _load_validation_selection(args.model_b_dir, model_name="Model B")

    model_a = load_validation_predictions(
        "Frozen ResNet-18 + Logistic Regression",
        args.model_a_dir / "validation_predictions.csv",
    )
    model_b_mean = load_validation_predictions(
        "Fine-tuned ResNet-18 mean pooling",
        args.model_b_dir / "validation_predictions_mean.csv",
    )
    model_b_max = load_validation_predictions(
        "Fine-tuned ResNet-18 max pooling (secondary diagnostic)",
        args.model_b_dir / "validation_predictions_max.csv",
    )
    predictions = [model_a, model_b_mean, model_b_max]
    assert_paired_cohort(predictions)
    _assert_model_b_run_matches_validation(
        args.model_b_dir, expected_studies=len(model_a.rows)
    )

    summaries = pd.DataFrame(
        [
            summarize_model(
                item,
                target=args.sensitivity_target,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
            for item in predictions
        ]
    )
    summaries.to_csv(output_dir / "model_comparison.csv", index=False)
    write_model_a_grouped_cv_summary(model_a_selection, output_dir)
    policy_columns = [
        "model",
        "sensitivity_target",
        "threshold_at_target",
        "sensitivity_at_target",
        "specificity_at_target",
        "precision_at_target",
        "f1_at_target",
        "false_negative_at_target",
        "false_positive_at_target",
        "flagged_percent_at_target",
    ]
    summaries[policy_columns].to_json(
        output_dir / "frozen_operating_policies.json", orient="records", indent=2
    )
    write_figures(predictions, summaries, output_dir, target=args.sensitivity_target)
    write_training_curve(args.model_b_dir / "training_history.json", output_dir)
    if args.shortcut_audit_dir is not None:
        write_shortcut_audit_summary(args.shortcut_audit_dir, output_dir)
    write_markdown(
        summaries,
        output_dir,
        args.sensitivity_target,
        include_shortcut_audit=args.shortcut_audit_dir is not None,
    )
    print(f"Wrote aggregate-only Checkpoint 1 results to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
