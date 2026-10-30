#!/usr/bin/env python
"""Train development-only frozen ResNet-18 + Logistic Regression baselines.

One independent study-level baseline is trained for each supported MURA
anatomy.  The command accepts manifests made by
``build_multi_anatomy_development_data.py`` only: they are derived from
official MURA ``train`` and contain a global patient-safe internal
train/validation split.  Official MURA ``valid`` is rejected and is never
opened by this script.

The model is intentionally modest:

    image -> frozen ImageNet ResNet-18 -> 512-D embedding
    views in one study -> mean embedding
    study embedding -> Logistic Regression

``C`` is selected with patient-grouped CV inside each anatomy's internal
training data.  The operating threshold is then selected only on that
anatomy's internal validation studies: the highest threshold that reaches the
pre-specified sensitivity target (95% by default).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.dataset import MURAImageDataset
from src.data.multi_anatomy import SUPPORTED_MURA_ANATOMIES
from src.data.transforms import build_eval_transform, preprocessing_spec
from src.features.extract_embeddings import (
    FrozenResNet18Embedder,
    StudyEmbeddings,
    extract_study_embeddings,
    save_study_embeddings,
)
from src.models.frozen_embedding_baseline import FrozenEmbeddingLogisticBaseline
from src.training.metrics import (
    compute_binary_study_metrics,
    select_highest_threshold_for_sensitivity,
)


DEFAULT_C_VALUES = (0.1, 1.0, 10.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests-dir", type=Path, required=True, help=".../manifests/by_anatomy")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Extracted MURA-v1.1 directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anatomies", nargs="+", choices=SUPPORTED_MURA_ANATOMIES, default=list(SUPPORTED_MURA_ANATOMIES))
    parser.add_argument("--target-sensitivity", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser


def resolve_device(requested: str) -> torch.device:
    requested = str(requested).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this Python environment.")
    return device


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return [{key: str(value or "") for key, value in row.items()} for row in rows]


def validate_anatomy_manifest(manifest_path: Path, anatomy: str) -> None:
    """Fail closed before image loading if a manifest has unsafe provenance."""

    rows = _read_csv(manifest_path)
    required = {"anatomy", "patient_id", "study_id", "image_path", "label", "source_split"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{manifest_path} is missing required fields: {sorted(missing)}")
    for row in rows:
        if row["anatomy"] != anatomy:
            raise ValueError(f"{manifest_path} contains an unexpected anatomy: {row['anatomy']!r}")
        if row["source_split"] != "train":
            raise ValueError("Only official MURA train rows are allowed in multi-anatomy development.")
        image_path = row["image_path"].replace("\\", "/")
        if image_path.startswith("valid/") or "/valid/" in image_path:
            raise ValueError("Official MURA valid/test paths are forbidden in this baseline command.")
        if not row["study_id"].startswith(f"{anatomy}/"):
            raise ValueError("Study IDs must include anatomy to prevent cross-anatomy view pooling.")


def embeddings_for_split(
    manifest_path: Path,
    split_path: Path,
    *,
    anatomy: str,
    final_split: str,
    dataset_root: Path,
    batch_size: int,
    workers: int,
    device: torch.device,
    embedder: FrozenResNet18Embedder,
) -> StudyEmbeddings:
    dataset = MURAImageDataset(
        manifest_path,
        dataset_root=dataset_root,
        split_manifest=split_path,
        split=final_split,
        transform=build_eval_transform(image_size=224),
    )
    if not dataset.records:
        raise RuntimeError(f"{anatomy} {final_split} has no image rows.")
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(workers),
    )
    result = extract_study_embeddings(loader, embedder, device=device)
    result.metadata.update(
        {
            "anatomy": anatomy,
            "final_split": final_split,
            "source_population": "official_mura_train_only",
            "official_valid_read": False,
        }
    )
    return result


def _metric_dict(labels: np.ndarray, probabilities: np.ndarray, *, threshold: float) -> dict[str, float | int | None]:
    metrics = compute_binary_study_metrics(labels, probabilities, threshold=threshold)
    predicted = (probabilities >= threshold).astype(int)
    return {
        **metrics.to_dict(),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "flagged_abnormal": int(predicted.sum()),
        "flagged_percent": float(100.0 * predicted.mean()),
    }


def patient_cluster_bootstrap(
    labels: np.ndarray,
    probabilities: np.ndarray,
    patient_ids: np.ndarray,
    *,
    threshold: float,
    replicates: int,
    seed: int,
) -> list[dict[str, float | str | int]]:
    """Patient-cluster bootstrap for fixed-validation-threshold metrics.

    The threshold has already been selected on this same validation set.  The
    resulting intervals quantify sampling variation conditional on that fixed
    chosen threshold; they do not make the validation set an untouched test.
    """

    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    patients = np.asarray(patient_ids, dtype=str)
    unique_patients = np.unique(patients)
    lookup = {patient: np.flatnonzero(patients == patient) for patient in unique_patients}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {key: [] for key in ("auroc", "auprc", "sensitivity", "specificity", "precision", "f1")}
    for _ in range(replicates):
        sampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        indices = np.concatenate([lookup[patient] for patient in sampled_patients])
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        values = _metric_dict(sampled_labels, probabilities[indices], threshold=threshold)
        for key in samples:
            value = values[key]
            if value is not None:
                samples[key].append(float(value))
    rows: list[dict[str, float | str | int]] = []
    for metric, values in samples.items():
        lower, upper = (np.quantile(values, [0.025, 0.975]).tolist() if values else (np.nan, np.nan))
        rows.append(
            {
                "metric": metric,
                "estimate": float(_metric_dict(labels, probabilities, threshold=threshold)[metric]),
                "lower_95_ci": float(lower),
                "upper_95_ci": float(upper),
                "valid_resamples": int(len(values)),
                "requested_resamples": int(replicates),
                "bootstrap_unit": "patient",
            }
        )
    return rows


def save_curves(predictions: dict[str, pd.DataFrame], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for anatomy, frame in predictions.items():
        labels = frame["label"].to_numpy(dtype=int)
        probabilities = frame["abnormal_probability"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(labels, probabilities)
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        axes[0].plot(fpr, tpr, linewidth=1.8, label=f"{anatomy.replace('XR_', '')} ({roc_auc_score(labels, probabilities):.3f})")
        axes[1].plot(recall, precision, linewidth=1.8, label=f"{anatomy.replace('XR_', '')} ({average_precision_score(labels, probabilities):.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1)
    axes[0].set(xlabel="False-positive rate", ylabel="Sensitivity", title="Internal validation ROC curves", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].set(xlabel="Sensitivity", ylabel="Precision", title="Internal validation precision–recall curves", xlim=(0, 1), ylim=(0, 1.02))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="lower right")
    figure.suptitle("Frozen ResNet-18 + Logistic Regression, one model per MURA anatomy", y=1.02, fontsize=13)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def save_operating_point_chart(metrics: pd.DataFrame, output_path: Path) -> None:
    display_names = metrics["anatomy"].str.replace("XR_", "", regex=False)
    positions = np.arange(len(metrics))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].bar(positions - 0.19, metrics["sensitivity"], width=0.38, label="Sensitivity", color="#0f766e")
    axes[0].bar(positions + 0.19, metrics["specificity"], width=0.38, label="Specificity", color="#2563eb")
    axes[0].axhline(0.95, color="#b91c1c", linestyle="--", linewidth=1.2, label="Target sensitivity")
    axes[0].set(xticks=positions, xticklabels=display_names, ylim=(0, 1.05), ylabel="Validation proportion", title="Operating point selected on internal validation")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(positions, metrics["flagged_percent"], color="#7c3aed")
    axes[1].set(xticks=positions, xticklabels=display_names, ylim=(0, 100), ylabel="Studies flagged abnormal (%)", title="Review workload at the selected operating point")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Path]:
    if not 0.0 < float(args.target_sensitivity) <= 1.0:
        raise ValueError("--target-sensitivity must be in (0, 1].")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("--batch-size must be positive and --workers must be non-negative.")
    manifests_dir = args.manifests_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    output = args.output_dir.resolve()
    if not manifests_dir.is_dir() or not dataset_root.is_dir():
        raise FileNotFoundError("The manifests directory and extracted MURA dataset root must both exist.")
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    # Load the frozen encoder once. Its weights stay fixed for all seven models.
    embedder = FrozenResNet18Embedder(weights="imagenet", allow_weight_download=False)
    predictions_by_anatomy: dict[str, pd.DataFrame] = {}
    metrics_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for anatomy in args.anatomies:
        print(f"\n[{anatomy}] extracting frozen study embeddings")
        stem = anatomy.lower()
        manifest_path = manifests_dir / f"{stem}_manifest.csv"
        split_path = manifests_dir / f"{stem}_split.csv"
        validate_anatomy_manifest(manifest_path, anatomy)
        train_embeddings = embeddings_for_split(
            manifest_path, split_path, anatomy=anatomy, final_split="train", dataset_root=dataset_root,
            batch_size=args.batch_size, workers=args.workers, device=device, embedder=embedder,
        )
        validation_embeddings = embeddings_for_split(
            manifest_path, split_path, anatomy=anatomy, final_split="val", dataset_root=dataset_root,
            batch_size=args.batch_size, workers=args.workers, device=device, embedder=embedder,
        )
        embedding_dir = output / "embeddings" / stem
        save_study_embeddings(train_embeddings, embedding_dir / "train_mean_embeddings.npz")
        save_study_embeddings(validation_embeddings, embedding_dir / "validation_mean_embeddings.npz")

        baseline = FrozenEmbeddingLogisticBaseline(
            c_values=DEFAULT_C_VALUES,
            selection_metric="auroc",
            class_weight=None,
            random_seed=int(args.seed),
        ).fit(
            train_embeddings.embeddings,
            train_embeddings.labels,
            train_embeddings.patient_ids,
            group_cv_folds=5,
        )
        validation_probabilities = baseline.predict_proba(validation_embeddings.embeddings)
        selection = select_highest_threshold_for_sensitivity(
            validation_embeddings.labels,
            validation_probabilities,
            target_sensitivity=float(args.target_sensitivity),
        )
        metrics = _metric_dict(
            validation_embeddings.labels,
            validation_probabilities,
            threshold=selection.threshold,
        )
        selection_metadata = baseline.selection_metadata()
        model_dir = output / "models" / stem
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(baseline, model_dir / "frozen_resnet18_logreg.joblib")
        (model_dir / "selection.json").write_text(
            json.dumps({
                "anatomy": anatomy,
                "model": "Frozen ImageNet ResNet-18 embeddings + Logistic Regression",
                "embedding_pooling": "mean_embedding",
                "C_selection": selection_metadata,
                "threshold_selection": selection.to_dict(),
                "selected_on": "internal_validation_only",
                "official_valid_read": False,
            }, indent=2) + "\n",
            encoding="utf-8",
        )

        prediction_frame = pd.DataFrame(
            {
                "anatomy": anatomy,
                "split": "internal_validation",
                "patient_id": validation_embeddings.patient_ids,
                "study_id": validation_embeddings.study_ids,
                "label": validation_embeddings.labels,
                "num_views": validation_embeddings.num_views,
                "abnormal_probability": validation_probabilities,
                "selected_threshold": selection.threshold,
                "predicted_abnormal": (validation_probabilities >= selection.threshold).astype(int),
            }
        ).sort_values("study_id", kind="stable")
        prediction_path = output / "validation_predictions" / f"{stem}_validation_predictions.csv"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_frame.to_csv(prediction_path, index=False)
        predictions_by_anatomy[anatomy] = prediction_frame

        metrics_rows.append(
            {
                "anatomy": anatomy,
                "model": "Frozen ResNet-18 embeddings + Logistic Regression",
                "evaluation_population": "internal_validation",
                "n_studies": metrics["n_studies"],
                "n_patients": int(pd.Series(validation_embeddings.patient_ids).nunique()),
                "normal_studies": metrics["n_normal"],
                "abnormal_studies": metrics["n_abnormal"],
                "selected_C": selection_metadata["selected_c"],
                "threshold": selection.threshold,
                "target_sensitivity": args.target_sensitivity,
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "false_negative_rate": metrics["false_negative_rate"],
                "false_positive": metrics["false_positive"],
                "false_negative": metrics["false_negative"],
                "true_positive": metrics["true_positive"],
                "true_negative": metrics["true_negative"],
                "flagged_abnormal": metrics["flagged_abnormal"],
                "flagged_percent": metrics["flagged_percent"],
                "official_valid_read": False,
            }
        )
        threshold_rows.append(
            {
                "anatomy": anatomy,
                **selection.to_dict(),
                "selection_population": "internal_validation",
                "official_valid_read": False,
            }
        )
        for bootstrap in patient_cluster_bootstrap(
            validation_embeddings.labels,
            validation_probabilities,
            validation_embeddings.patient_ids,
            threshold=selection.threshold,
            replicates=int(args.bootstrap_replicates),
            seed=int(args.seed),
        ):
            bootstrap_rows.append({"anatomy": anatomy, **bootstrap})
        print(
            f"[{anatomy}] AUROC={metrics['auroc']:.3f}, AUPRC={metrics['auprc']:.3f}, "
            f"threshold={selection.threshold:.4f}, sensitivity={metrics['sensitivity']:.3f}, "
            f"specificity={metrics['specificity']:.3f}"
        )

    metrics_frame = pd.DataFrame(metrics_rows).sort_values("anatomy", kind="stable")
    thresholds_frame = pd.DataFrame(threshold_rows).sort_values("anatomy", kind="stable")
    bootstrap_frame = pd.DataFrame(bootstrap_rows).sort_values(["anatomy", "metric"], kind="stable")
    metrics_path = output / "baseline_metrics_by_anatomy.csv"
    thresholds_path = output / "baseline_thresholds_by_anatomy.csv"
    bootstrap_path = output / "baseline_patient_bootstrap_ci.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    thresholds_frame.to_csv(thresholds_path, index=False)
    bootstrap_frame.to_csv(bootstrap_path, index=False)
    curves_path = output / "baseline_roc_pr_by_anatomy.png"
    workload_path = output / "baseline_operating_point_workload.png"
    save_curves(predictions_by_anatomy, curves_path)
    save_operating_point_chart(metrics_frame, workload_path)

    provenance = {
        "scope": "development-only MURA all-anatomy baseline",
        "anatomies": list(args.anatomies),
        "model": "Frozen ImageNet ResNet-18 embeddings + mean study embedding pooling + Logistic Regression",
        "C_values": list(DEFAULT_C_VALUES),
        "C_selection": "5-fold patient-grouped CV inside each anatomy internal train only",
        "operating_policy": "highest validation threshold at sensitivity >= target",
        "target_sensitivity": float(args.target_sensitivity),
        "evaluation_population": "global patient-safe internal validation, reported per anatomy",
        "official_valid_read": False,
        "test_used_for_selection": False,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "workers": int(args.workers),
        "seed": int(args.seed),
        "bootstrap": {
            "unit": "patient",
            "replicates": int(args.bootstrap_replicates),
            "note": "Intervals are conditional on thresholds selected from the same validation set; this is development evidence, not an untouched test result.",
        },
        "preprocessing": preprocessing_spec(),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    provenance_path = output / "baseline_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("\nDevelopment-only summary (official MURA valid was not read):")
    print(metrics_frame[["anatomy", "n_studies", "selected_C", "threshold", "auroc", "auprc", "sensitivity", "specificity", "flagged_percent"]].to_string(index=False))
    return {
        "metrics": metrics_path,
        "thresholds": thresholds_path,
        "bootstrap": bootstrap_path,
        "curves": curves_path,
        "workload": workload_path,
        "provenance": provenance_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outputs = run(args)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
