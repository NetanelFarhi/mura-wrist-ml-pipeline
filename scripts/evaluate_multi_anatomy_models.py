#!/usr/bin/env python
"""Compare all-anatomy frozen and fine-tuned models on internal validation.

This is a local, development-only report. It refuses any artifact that claims
official MURA valid/test use. The two models are compared per anatomy on the
same patient-safe internal validation studies, each at its own pre-specified
95% sensitivity operating policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, roc_auc_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.multi_anatomy import SUPPORTED_MURA_ANATOMIES
from src.training.metrics import compute_binary_study_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--finetuned-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def read_csv_round_trip(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, float_precision="round_trip")


def load_prediction_pair(baseline_dir: Path, finetuned_dir: Path, anatomy: str) -> tuple[pd.DataFrame, float, float]:
    stem = anatomy.lower()
    baseline = read_csv_round_trip(baseline_dir / "validation_predictions" / f"{stem}_validation_predictions.csv")
    finetuned = read_csv_round_trip(finetuned_dir / stem / "validation_predictions.csv")
    baseline_thresholds = read_csv_round_trip(baseline_dir / "baseline_thresholds_by_anatomy.csv")
    selected_baseline_threshold = baseline_thresholds.loc[baseline_thresholds["anatomy"] == anatomy, "threshold"]
    if len(selected_baseline_threshold) != 1:
        raise ValueError(f"Expected exactly one frozen baseline threshold for {anatomy}.")
    baseline_threshold = float(selected_baseline_threshold.iloc[0])
    selection = json.loads((finetuned_dir / stem / "selection.json").read_text(encoding="utf-8"))
    if selection.get("test_used_for_selection") is not False:
        raise ValueError("Fine-tuned selection provenance does not rule out test use.")
    finetuned_threshold = float(selection["threshold_selection"]["threshold"])
    required = {"anatomy", "split", "patient_id", "study_id", "label", "num_views", "abnormal_probability"}
    for name, frame in (("baseline", baseline), ("fine-tuned", finetuned)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} predictions for {anatomy} are missing {sorted(missing)}")
        if not frame["anatomy"].eq(anatomy).all() or not frame["split"].eq("internal_validation").all():
            raise ValueError(f"{name} predictions for {anatomy} are not exclusively internal-validation rows.")
        if frame["study_id"].duplicated().any() or not frame["study_id"].str.startswith(f"{anatomy}/").all():
            raise ValueError(f"{name} predictions do not have one valid study row per {anatomy} study.")
    baseline = baseline.set_index("study_id").sort_index()
    finetuned = finetuned.set_index("study_id").sort_index()
    if not baseline.index.equals(finetuned.index):
        raise ValueError(f"Model predictions are not paired on the same {anatomy} validation studies.")
    for field in ("patient_id", "label", "num_views"):
        if not baseline[field].equals(finetuned[field]):
            raise ValueError(f"{anatomy} models disagree on paired field {field!r}")
    return baseline.reset_index(), baseline_threshold, finetuned_threshold


def report_metrics(rows: pd.DataFrame, threshold: float) -> dict[str, float | int | None]:
    labels = rows["label"].to_numpy(dtype=int)
    probabilities = rows["abnormal_probability"].to_numpy(dtype=float)
    metrics = compute_binary_study_metrics(labels, probabilities, threshold=threshold).to_dict()
    predicted = probabilities >= threshold
    metrics.update(
        {
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "flagged_abnormal": int(predicted.sum()),
            "flagged_percent": float(100 * predicted.mean()),
        }
    )
    return metrics


def patient_paired_bootstrap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    baseline_threshold: float,
    candidate_threshold: float,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Paired patient-cluster CIs for candidate minus baseline differences."""

    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    if not baseline["patient_id"].equals(candidate["patient_id"]):
        raise ValueError("Paired bootstrap requires aligned patient rows.")
    labels = baseline["label"].to_numpy(dtype=int)
    patients = baseline["patient_id"].astype(str).to_numpy()
    base_scores = baseline["abnormal_probability"].to_numpy(dtype=float)
    candidate_scores = candidate["abnormal_probability"].to_numpy(dtype=float)
    unique_patients = np.unique(patients)
    index_by_patient = {patient: np.flatnonzero(patients == patient) for patient in unique_patients}
    metric_names = ("auroc", "auprc", "sensitivity", "specificity", "precision", "f1", "flagged_percent")
    differences = {name: [] for name in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        drawn = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        indices = np.concatenate([index_by_patient[patient] for patient in drawn])
        if len(np.unique(labels[indices])) < 2:
            continue
        base = pd.DataFrame({"label": labels[indices], "abnormal_probability": base_scores[indices]})
        cand = pd.DataFrame({"label": labels[indices], "abnormal_probability": candidate_scores[indices]})
        base_report = report_metrics(base, baseline_threshold)
        candidate_report = report_metrics(cand, candidate_threshold)
        for name in metric_names:
            differences[name].append(float(candidate_report[name]) - float(base_report[name]))
    base_report = report_metrics(baseline, baseline_threshold)
    candidate_report = report_metrics(candidate, candidate_threshold)
    rows = []
    for name, values in differences.items():
        lower, upper = np.quantile(values, [0.025, 0.975]).tolist()
        rows.append(
            {
                "metric": name,
                "candidate_minus_baseline": float(candidate_report[name]) - float(base_report[name]),
                "lower_95_ci": float(lower),
                "upper_95_ci": float(upper),
                "valid_resamples": len(values),
                "requested_resamples": replicates,
                "bootstrap_unit": "patient",
            }
        )
    return pd.DataFrame(rows)


def save_figures(comparison: pd.DataFrame, histories: pd.DataFrame, output_dir: Path) -> None:
    summary = comparison.pivot(index="anatomy", columns="model", values=["auroc", "auprc", "specificity", "flagged_percent"])
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.0))
    auroc = summary["auroc"]
    auroc.plot(kind="bar", ax=axes[0], color=["#64748b", "#2563eb"])
    axes[0].set(title="Internal validation AUROC", xlabel="Anatomy", ylabel="AUROC", ylim=(0, 1.05))
    axes[0].grid(axis="y", alpha=0.25)
    specificity = summary["specificity"]
    specificity.plot(kind="bar", ax=axes[1], color=["#64748b", "#0f766e"])
    axes[1].set(title="Specificity at the 95% sensitivity policy", xlabel="Anatomy", ylabel="Specificity", ylim=(0, 1.05))
    axes[1].grid(axis="y", alpha=0.25)
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
        axis.legend(fontsize=8)
    figure.suptitle("Development-only comparison: frozen baseline vs fine-tuned ResNet-18", y=1.02, fontsize=13)
    figure.tight_layout()
    figure.savefig(output_dir / "baseline_vs_finetuned.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharey=False)
    for axis, anatomy in zip(axes.flat, SUPPORTED_MURA_ANATOMIES):
        subset = histories.loc[histories["anatomy"] == anatomy].sort_values("epoch")
        axis.plot(subset["epoch"], subset["validation_auroc"], marker="o", color="#2563eb", label="Validation AUROC")
        axis.set(title=anatomy.replace("XR_", ""), xlabel="Epoch", ylabel="AUROC", ylim=(0, 1.05))
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    figure.suptitle("Fine-tuning validation AUROC by epoch", y=0.995, fontsize=13)
    figure.tight_layout()
    figure.savefig(output_dir / "finetuned_training_history.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Path]:
    baseline_dir = args.baseline_dir.resolve()
    finetuned_dir = args.finetuned_dir.resolve()
    output = args.output_dir.resolve()
    baseline_provenance = json.loads((baseline_dir / "baseline_provenance.json").read_text(encoding="utf-8"))
    fine_provenance = json.loads((finetuned_dir / "finetuned_provenance.json").read_text(encoding="utf-8"))
    if baseline_provenance.get("official_valid_read") is not False or fine_provenance.get("official_valid_read") is not False:
        raise ValueError("This local comparison requires train-only development artifacts.")
    output.mkdir(parents=True, exist_ok=True)
    comparison_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for anatomy in SUPPORTED_MURA_ANATOMIES:
        baseline, baseline_threshold, candidate_threshold = load_prediction_pair(baseline_dir, finetuned_dir, anatomy)
        candidate = read_csv_round_trip(finetuned_dir / anatomy.lower() / "validation_predictions.csv")
        candidate = candidate.set_index("study_id").loc[baseline.set_index("study_id").index].reset_index()
        baseline_report = report_metrics(baseline, baseline_threshold)
        candidate_report = report_metrics(candidate, candidate_threshold)
        for model, threshold, report in (
            ("Frozen ResNet-18 + Logistic Regression", baseline_threshold, baseline_report),
            ("Fine-tuned ResNet-18", candidate_threshold, candidate_report),
        ):
            comparison_rows.append(
                {
                    "anatomy": anatomy,
                    "model": model,
                    "evaluation_population": "internal_validation",
                    "n_studies": report["n_studies"],
                    "threshold": threshold,
                    "auroc": report["auroc"],
                    "auprc": report["auprc"],
                    "sensitivity": report["sensitivity"],
                    "specificity": report["specificity"],
                    "precision": report["precision"],
                    "f1": report["f1"],
                    "false_negative": report["false_negative"],
                    "false_positive": report["false_positive"],
                    "flagged_percent": report["flagged_percent"],
                    "official_valid_read": False,
                }
            )
        paired = patient_paired_bootstrap(
            baseline, candidate, baseline_threshold=baseline_threshold, candidate_threshold=candidate_threshold,
            replicates=int(args.bootstrap_replicates), seed=int(args.seed),
        )
        paired.insert(0, "anatomy", anatomy)
        paired_rows.extend(paired.to_dict(orient="records"))

        baseline_pred = baseline["abnormal_probability"].to_numpy(dtype=float) >= baseline_threshold
        candidate_pred = candidate["abnormal_probability"].to_numpy(dtype=float) >= candidate_threshold
        labels = baseline["label"].to_numpy(dtype=int)
        error_rows.append(
            {
                "anatomy": anatomy,
                "both_correct": int(np.sum((baseline_pred == labels) & (candidate_pred == labels))),
                "both_incorrect": int(np.sum((baseline_pred != labels) & (candidate_pred != labels))),
                "errors_fixed_by_finetuning": int(np.sum((baseline_pred != labels) & (candidate_pred == labels))),
                "errors_introduced_by_finetuning": int(np.sum((baseline_pred == labels) & (candidate_pred != labels))),
                "false_negatives_fixed": int(np.sum((labels == 1) & (baseline_pred == 0) & (candidate_pred == 1))),
                "false_negatives_introduced": int(np.sum((labels == 1) & (baseline_pred == 1) & (candidate_pred == 0))),
            }
        )
        history = json.loads((finetuned_dir / anatomy.lower() / "training_history.json").read_text(encoding="utf-8"))
        for epoch in history["history"]:
            history_rows.append(
                {
                    "anatomy": anatomy,
                    "epoch": epoch["epoch"],
                    "train_loss": epoch["train_loss"],
                    "validation_auroc": epoch["validation_monitor"],
                    "learning_rate": epoch["learning_rate"],
                    "best_epoch": history["best_epoch"],
                }
            )
    comparison = pd.DataFrame(comparison_rows).sort_values(["anatomy", "model"], kind="stable")
    paired = pd.DataFrame(paired_rows).sort_values(["anatomy", "metric"], kind="stable")
    errors = pd.DataFrame(error_rows).sort_values("anatomy", kind="stable")
    histories = pd.DataFrame(history_rows).sort_values(["anatomy", "epoch"], kind="stable")
    comparison_path = output / "model_comparison_by_anatomy.csv"
    paired_path = output / "paired_patient_bootstrap.csv"
    errors_path = output / "paired_error_summary.csv"
    history_path = output / "training_history_by_anatomy.csv"
    comparison.to_csv(comparison_path, index=False)
    paired.to_csv(paired_path, index=False)
    errors.to_csv(errors_path, index=False)
    histories.to_csv(history_path, index=False)
    save_figures(comparison, histories, output)
    summary = {
        "scope": "development-only paired all-anatomy model comparison",
        "evaluation_population": "same internal validation studies per anatomy",
        "operating_policy": "each model uses its own highest validation threshold at sensitivity >= 0.95",
        "official_valid_read": False,
        "test_used_for_selection": False,
        "bootstrap": {"unit": "patient", "replicates": int(args.bootstrap_replicates), "comparison": "fine-tuned minus frozen baseline"},
        "interpretation": "Confidence intervals are conditional on validation-selected thresholds and do not constitute untouched-test evidence.",
    }
    summary_path = output / "comparison_provenance.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(comparison[["anatomy", "model", "auroc", "auprc", "sensitivity", "specificity", "false_negative", "false_positive"]].to_string(index=False))
    return {"comparison": comparison_path, "paired": paired_path, "errors": errors_path, "history": history_path, "provenance": summary_path}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outputs = run(args)
    except (FileNotFoundError, ValueError, RuntimeError, OSError, IndexError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
