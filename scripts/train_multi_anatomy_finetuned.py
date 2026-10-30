#!/usr/bin/env python
"""Train one development-only study-level fine-tuned ResNet-18 per anatomy.

This wrapper adds the all-anatomy safety contract around the existing
study-level trainer.  It trains a separate model for each requested MURA
anatomy, using only its official-train-derived manifest and the global
patient-safe internal split.  It rejects official MURA ``valid`` paths and
does not expose an option to create test predictions.

The controlled recipe is deliberately fixed before the results are seen:
ImageNet ResNet-18, study-level mean pooling of view probabilities, AdamW,
maximum 15 epochs, validation-AUROC early stopping, and a per-anatomy
internal-validation operating threshold at sensitivity >= 95%.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# Must be set before PyTorch initializes CUDA when deterministic algorithms
# are requested by the shared study-level trainer.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from src.data.multi_anatomy import SUPPORTED_MURA_ANATOMIES
from src.training.metrics import compute_binary_study_metrics
from scripts import train_resnet18


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests-dir", type=Path, required=True, help=".../manifests/by_anatomy")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anatomies", nargs="+", choices=SUPPORTED_MURA_ANATOMIES, default=list(SUPPORTED_MURA_ANATOMIES))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--encoder-weights", choices=("imagenet",), default="imagenet")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--sensitivity-target", type=float, default=0.95)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return [{key: str(value or "") for key, value in row.items()} for row in rows]


def validate_development_inputs(manifest_path: Path, split_path: Path, anatomy: str) -> None:
    """Reject unsafe/ambiguous sources before any image decoding or training."""

    manifest_rows = _read_csv(manifest_path)
    split_rows = _read_csv(split_path)
    required_manifest = {"anatomy", "patient_id", "study_id", "image_path", "label", "source_split", "num_views_in_study"}
    required_split = {"patient_id", "study_id", "final_split"}
    if missing := required_manifest.difference(manifest_rows[0]):
        raise ValueError(f"Manifest missing fields: {sorted(missing)}")
    if missing := required_split.difference(split_rows[0]):
        raise ValueError(f"Split manifest missing fields: {sorted(missing)}")
    studies: dict[tuple[str, str], dict[str, str]] = {}
    for row in manifest_rows:
        image_path = row["image_path"].replace("\\", "/")
        if row["anatomy"] != anatomy:
            raise ValueError(f"Expected {anatomy}, found {row['anatomy']!r} in {manifest_path.name}")
        if row["source_split"] != "train":
            raise ValueError("Fine-tuning accepts only official MURA train-derived rows.")
        if image_path.startswith("valid/") or "/valid/" in image_path:
            raise ValueError("Official MURA valid/test paths are forbidden.")
        if not row["study_id"].startswith(f"{anatomy}/"):
            raise ValueError("Every study_id must include the declared anatomy.")
        if row["label"] not in {"0", "1"}:
            raise ValueError("Labels must be binary 0/1.")
        key = (row["patient_id"], row["study_id"])
        current = studies.setdefault(key, row)
        if (current["label"], current["anatomy"], current["num_views_in_study"]) != (
            row["label"], row["anatomy"], row["num_views_in_study"]
        ):
            raise ValueError(f"Inconsistent study metadata: {row['study_id']}")
    assignments: dict[tuple[str, str], str] = {}
    patient_splits: dict[str, set[str]] = {}
    for row in split_rows:
        final_split = row["final_split"].strip().lower()
        if final_split not in {"train", "val"}:
            raise ValueError("The multi-anatomy development split may contain only train and val, never test.")
        key = (row["patient_id"], row["study_id"])
        if key not in studies:
            raise ValueError(f"Split references a study absent from the anatomy manifest: {row['study_id']}")
        if key in assignments:
            raise ValueError(f"Duplicate split assignment: {row['study_id']}")
        assignments[key] = final_split
        patient_splits.setdefault(row["patient_id"], set()).add(final_split)
    if set(assignments) != set(studies):
        raise ValueError("Every anatomy study must have exactly one internal train/validation assignment.")
    leaking_patients = [patient for patient, splits in patient_splits.items() if len(splits) != 1]
    if leaking_patients:
        raise ValueError(f"Patient leakage across internal splits: {leaking_patients[:3]}")


def patient_cluster_bootstrap(
    rows: pd.DataFrame, *, threshold: float, replicates: int, seed: int
) -> pd.DataFrame:
    """Patient-cluster intervals for validation metrics at a frozen threshold."""

    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    labels = rows["label"].to_numpy(dtype=int)
    probabilities = rows["abnormal_probability"].to_numpy(dtype=float)
    patients = rows["patient_id"].astype(str).to_numpy()
    unique_patients = np.unique(patients)
    index_by_patient = {patient: np.flatnonzero(patients == patient) for patient in unique_patients}
    metrics = {name: [] for name in ("auroc", "auprc", "sensitivity", "specificity", "precision", "f1")}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        drawn = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        indices = np.concatenate([index_by_patient[patient] for patient in drawn])
        sample_labels = labels[indices]
        if len(np.unique(sample_labels)) < 2:
            continue
        report = compute_binary_study_metrics(sample_labels, probabilities[indices], threshold=threshold).to_dict()
        predicted = probabilities[indices] >= threshold
        report["precision"] = float(np.sum((sample_labels == 1) & predicted) / max(int(predicted.sum()), 1))
        for name in metrics:
            metrics[name].append(float(report[name]))
    point = compute_binary_study_metrics(labels, probabilities, threshold=threshold).to_dict()
    predicted = probabilities >= threshold
    point["precision"] = float(np.sum((labels == 1) & predicted) / max(int(predicted.sum()), 1))
    output_rows = []
    for name, values in metrics.items():
        lower, upper = np.quantile(values, [0.025, 0.975]).tolist()
        output_rows.append(
            {
                "metric": name,
                "estimate": float(point[name]),
                "lower_95_ci": float(lower),
                "upper_95_ci": float(upper),
                "valid_resamples": len(values),
                "requested_resamples": int(replicates),
                "bootstrap_unit": "patient",
            }
        )
    return pd.DataFrame(output_rows)


def train_one(args: argparse.Namespace, anatomy: str) -> dict[str, Any]:
    manifests_dir = args.manifests_dir.resolve()
    manifest_path = manifests_dir / f"{anatomy.lower()}_manifest.csv"
    split_path = manifests_dir / f"{anatomy.lower()}_split.csv"
    validate_development_inputs(manifest_path, split_path, anatomy)
    anatomy_output = args.output_dir.resolve() / anatomy.lower()
    anatomy_output.mkdir(parents=True, exist_ok=True)
    delegated = argparse.Namespace(
        manifest=manifest_path,
        split_manifest=split_path,
        output_dir=anatomy_output,
        dataset_root=args.dataset_root.resolve(),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        image_size=224,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        gradient_clip_norm=None,
        early_stopping_patience=int(args.early_stopping_patience),
        min_delta=0.0,
        seed=int(args.seed),
        device=args.device,
        weights=args.encoder_weights,
        allow_weight_download=False,
        class_weighting="auto",
        imbalance_ratio_threshold=1.5,
        train_aggregation="mean",
        monitor_metric="auroc",
        monitor_aggregation="mean",
        scheduler="plateau",
        scheduler_factor=0.5,
        scheduler_patience=2,
        min_learning_rate=1e-6,
        sensitivity_target=float(args.sensitivity_target),
        bootstrap_samples=int(args.bootstrap_replicates),
        write_test_predictions=False,
    )
    started = time.perf_counter()
    train_resnet18.run(delegated)
    runtime = time.perf_counter() - started

    prediction_paths = [anatomy_output / "validation_predictions.csv", anatomy_output / "validation_predictions_mean.csv", anatomy_output / "validation_predictions_max.csv"]
    for prediction_path in prediction_paths:
        # Pandas' default CSV parser may round a boundary probability by one
        # floating-point unit. ``round_trip`` keeps a serialized selected
        # threshold and its matching observed probability comparable.
        predictions = pd.read_csv(prediction_path, float_precision="round_trip")
        if predictions["study_id"].duplicated().any():
            raise RuntimeError("Fine-tuned prediction output contains duplicate study IDs.")
        if not predictions["study_id"].str.startswith(f"{anatomy}/").all():
            raise RuntimeError("Fine-tuned prediction output contains an unexpected anatomy study ID.")
        predictions.insert(0, "anatomy", anatomy)
        predictions.insert(1, "split", "internal_validation")
        # The selected threshold can be one of the observed probabilities.
        # Preserve enough digits to avoid changing a boundary case after the
        # table is written and read back for the report.
        predictions.to_csv(prediction_path, index=False, float_format="%.17g")

    predictions = pd.read_csv(
        anatomy_output / "validation_predictions.csv", float_precision="round_trip"
    )
    selection = json.loads((anatomy_output / "selection.json").read_text(encoding="utf-8"))
    threshold = float(selection["threshold_selection"]["threshold"])
    intervals = patient_cluster_bootstrap(
        predictions,
        threshold=threshold,
        replicates=int(args.bootstrap_replicates),
        seed=int(args.seed),
    )
    intervals.insert(0, "anatomy", anatomy)
    intervals.to_csv(anatomy_output / "patient_bootstrap_ci.csv", index=False)
    metrics = compute_binary_study_metrics(
        predictions["label"].to_numpy(dtype=int),
        predictions["abnormal_probability"].to_numpy(dtype=float),
        threshold=threshold,
    ).to_dict()
    if metrics["sensitivity"] is None or metrics["sensitivity"] + 1e-12 < float(args.sensitivity_target):
        raise RuntimeError(
            "The persisted validation predictions no longer meet the selected sensitivity target. "
            "Do not report or reuse this operating threshold."
        )
    predicted = predictions["abnormal_probability"].to_numpy(dtype=float) >= threshold
    metrics.update(
        {
            "precision": float(np.sum((predictions["label"].to_numpy(dtype=int) == 1) & predicted) / max(int(predicted.sum()), 1)),
            "flagged_abnormal": int(predicted.sum()),
            "flagged_percent": float(100 * predicted.mean()),
        }
    )
    summary = {
        "anatomy": anatomy,
        "model": "Fine-tuned ImageNet ResNet-18 with study-level mean probability pooling",
        "evaluation_population": "internal_validation",
        "n_studies": int(len(predictions)),
        "n_patients": int(predictions["patient_id"].nunique()),
        "selected_threshold": threshold,
        "target_sensitivity": float(args.sensitivity_target),
        "metrics": metrics,
        "official_valid_read": False,
        "test_used_for_selection": False,
        "runtime_seconds": round(runtime, 3),
    }
    (anatomy_output / "multi_anatomy_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    run_config_path = anatomy_output / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_config["multi_anatomy"] = {
        "anatomy": anatomy,
        "source_population": "official_mura_train_only",
        "official_valid_read": False,
        "test_used_for_selection": False,
        "study_id_contract": "anatomy/patient_id/study_directory",
    }
    run_config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    print(
        f"[{anatomy}] AUROC={metrics['auroc']:.3f}, AUPRC={metrics['auprc']:.3f}, "
        f"threshold={threshold:.4f}, sensitivity={metrics['sensitivity']:.3f}, "
        f"specificity={metrics['specificity']:.3f}"
    )
    return summary


def run(args: argparse.Namespace) -> Path:
    if not args.dataset_root.is_dir() or not args.manifests_dir.is_dir():
        raise FileNotFoundError("--dataset-root and --manifests-dir must exist.")
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("epochs/batch-size/workers are invalid.")
    if not 0.0 < args.sensitivity_target <= 1.0:
        raise ValueError("--sensitivity-target must be in (0, 1].")
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive.")
    rows: list[dict[str, Any]] = []
    for anatomy in args.anatomies:
        print(f"\n[{anatomy}] starting controlled study-level fine-tuning")
        summary = train_one(args, anatomy)
        metrics = summary["metrics"]
        rows.append(
            {
                "anatomy": anatomy,
                "model": summary["model"],
                "evaluation_population": summary["evaluation_population"],
                "n_studies": summary["n_studies"],
                "n_patients": summary["n_patients"],
                "threshold": summary["selected_threshold"],
                "target_sensitivity": summary["target_sensitivity"],
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "false_negative": metrics["false_negative"],
                "false_positive": metrics["false_positive"],
                "flagged_percent": metrics["flagged_percent"],
                "runtime_seconds": summary["runtime_seconds"],
                "official_valid_read": False,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "finetuned_metrics_by_anatomy.csv"
    pd.DataFrame(rows).sort_values("anatomy", kind="stable").to_csv(summary_path, index=False)
    provenance = {
        "scope": "development-only all-anatomy fine-tuned ResNet-18",
        "anatomies": list(args.anatomies),
        "initialization": "ImageNet ResNet-18; no historical wrist checkpoint reused",
        "architecture": "per-view ResNet-18 probabilities mean-pooled to one study prediction",
        "max_epochs": int(args.epochs),
        "early_stopping": {"monitor": "validation AUROC", "patience": int(args.early_stopping_patience)},
        "operating_policy": "highest validation threshold at sensitivity >= target",
        "target_sensitivity": float(args.sensitivity_target),
        "official_valid_read": False,
        "test_used_for_selection": False,
        "seed": int(args.seed),
    }
    (args.output_dir / "finetuned_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return summary_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(f"Saved all-anatomy fine-tuning summary: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
