#!/usr/bin/env python
"""Fine-tune Phase 1 ResNet-18 with one prediction and loss per MURA study.

By default this trains on the patient-safe train split and uses validation only
for early stopping, mean-vs-max aggregation selection, and threshold selection.
The final test split remains untouched unless ``--write-test-predictions`` is
passed; it then writes prediction rows only after those choices are saved.
``evaluate_phase1.py`` is the sole owner of final test metrics/comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.collate import collate_studies
from src.data.study_dataset import MURAStudyDataset
from src.data.transforms import build_eval_transform, build_train_transform, preprocessing_spec
from src.models.resnet18_classifier import (
    PretrainedWeightsUnavailableError,
    build_resnet18_classifier,
)
from src.training.losses import decide_class_weighting, make_study_probability_loss
from src.training.metrics import (
    bootstrap_auroc_ci,
    metrics_at_default_and_selected_threshold,
    select_operating_threshold_on_validation,
)
from src.training.trainer import (
    StudyLevelTrainer,
    StudyPredictions,
    make_data_loader_generator,
    resolve_device,
    save_json,
    seed_data_loader_worker,
    select_aggregation_on_validation,
    set_global_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Combined XR_WRIST image manifest CSV.")
    parser.add_argument("--split-manifest", type=Path, required=True, help="Patient-safe split manifest CSV.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="ImageNet initialization is required for Phase 1; 'none' only supports a smoke test.",
    )
    parser.add_argument(
        "--allow-weight-download",
        action="store_true",
        help="Explicitly permit downloading official torchvision ImageNet weights when uncached.",
    )
    parser.add_argument(
        "--class-weighting",
        choices=("none", "auto", "balanced"),
        default="auto",
        help="Uses training-study counts only; auto requires the configured imbalance threshold.",
    )
    parser.add_argument("--imbalance-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--train-aggregation", choices=("mean", "max"), default="mean")
    parser.add_argument("--monitor-metric", choices=("auroc", "auprc"), default="auroc")
    parser.add_argument("--monitor-aggregation", choices=("mean", "max"), default="mean")
    parser.add_argument("--scheduler", choices=("plateau", "none"), default="plateau")
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--threshold-objective",
        choices=("youden_j", "f1", "balanced_accuracy"),
        default="youden_j",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--write-test-predictions",
        "--evaluate-test",
        dest="write_test_predictions",
        action="store_true",
        help=(
            "After validation choices freeze, write test_predictions.csv only. "
            "Final test metrics/comparison are deferred to evaluate_phase1.py; "
            "--evaluate-test is retained as a legacy alias."
        ),
    )
    return parser


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty prediction table.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _dataset_study_rows(dataset: MURAStudyDataset) -> list[tuple[str, str, int]]:
    """Read metadata without decoding image pixels or loading test images."""

    studies = getattr(dataset, "studies", None)
    if studies is None:
        raise TypeError("Expected MURAStudyDataset with a .studies metadata collection.")
    result = [(str(item.patient_id), str(item.study_id), int(item.label)) for item in studies]
    if not result:
        raise ValueError("Requested split contains no studies.")
    return result


def _assert_disjoint_datasets(named_datasets: Sequence[tuple[str, MURAStudyDataset]]) -> None:
    metadata = [(name, _dataset_study_rows(dataset)) for name, dataset in named_datasets]
    for index, (left_name, left_rows) in enumerate(metadata):
        left_patients = {patient_id for patient_id, _, _ in left_rows}
        left_studies = {study_id for _, study_id, _ in left_rows}
        for right_name, right_rows in metadata[index + 1 :]:
            patient_overlap = left_patients.intersection(patient_id for patient_id, _, _ in right_rows)
            if patient_overlap:
                raise ValueError(
                    f"Patient leakage between {left_name} and {right_name}: {sorted(patient_overlap)[:3]}"
                )
            study_overlap = left_studies.intersection(study_id for _, study_id, _ in right_rows)
            if study_overlap:
                raise ValueError(
                    f"Study leakage between {left_name} and {right_name}: {sorted(study_overlap)[:3]}"
                )


def _make_loader(
    dataset: MURAStudyDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_studies,
        worker_init_fn=seed_data_loader_worker if workers else None,
        generator=make_data_loader_generator(seed),
        pin_memory=pin_memory,
    )


def _prediction_report(
    predictions: StudyPredictions,
    aggregation: str,
    *,
    threshold: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    probabilities = predictions.probabilities_by_aggregation[aggregation]
    return {
        "aggregation": aggregation,
        "metrics": metrics_at_default_and_selected_threshold(
            predictions.labels, probabilities, threshold
        ),
        "auroc_bootstrap_ci": bootstrap_auroc_ci(
            predictions.labels, probabilities, n_bootstrap=bootstrap_samples, seed=seed
        ),
    }


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must both be at least 1.")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("--learning-rate must be positive and --weight-decay cannot be negative.")
    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be at least 1.")
    if args.min_delta < 0:
        raise ValueError("--min-delta cannot be negative.")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be at least 1.")
    if args.scheduler == "plateau":
        if not 0 < args.scheduler_factor < 1:
            raise ValueError("--scheduler-factor must be between 0 and 1.")
        if args.scheduler_patience < 0 or args.min_learning_rate < 0:
            raise ValueError("Scheduler patience/minimum learning rate must be non-negative.")


def run(args: argparse.Namespace) -> dict[str, Path]:
    _validate_arguments(args)
    set_global_seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    train_dataset = MURAStudyDataset(
        args.manifest,
        dataset_root=args.dataset_root,
        split_manifest=args.split_manifest,
        split="train",
        transform=build_train_transform(image_size=args.image_size, seed=args.seed),
    )
    validation_dataset = MURAStudyDataset(
        args.manifest,
        dataset_root=args.dataset_root,
        split_manifest=args.split_manifest,
        split="val",
        transform=build_eval_transform(image_size=args.image_size),
    )
    _assert_disjoint_datasets((("train", train_dataset), ("validation", validation_dataset)))
    train_metadata = _dataset_study_rows(train_dataset)
    train_labels = np.asarray([label for _, _, label in train_metadata], dtype=np.int64)
    class_weight_decision = decide_class_weighting(
        train_labels,
        mode=args.class_weighting,
        imbalance_ratio_threshold=args.imbalance_ratio_threshold,
    )
    train_loader = _make_loader(
        train_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=True,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    validation_loader = _make_loader(
        validation_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )

    model = build_resnet18_classifier(
        weights=args.weights,
        allow_weight_download=args.allow_weight_download,
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.min_learning_rate,
        )
    trainer = StudyLevelTrainer(
        model,
        optimizer,
        make_study_probability_loss(class_weight_decision, device=device),
        scheduler=scheduler,
        device=device,
        train_aggregation=args.train_aggregation,
        gradient_clip_norm=args.gradient_clip_norm,
    )
    checkpoint_path = output_dir / "best_resnet18_study_level.pt"
    training_result = trainer.fit(
        train_loader,
        validation_loader,
        max_epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_path=checkpoint_path,
        monitor_metric=args.monitor_metric,
        monitor_aggregation=args.monitor_aggregation,
        min_delta=args.min_delta,
    )
    save_json(training_result, output_dir / "training_history.json")

    # The restored model is the validation-best checkpoint.  Both required
    # aggregations are compared on validation exactly once after training.
    validation_predictions = trainer.predict(validation_loader, aggregations=("mean", "max"))
    aggregation_selection = select_aggregation_on_validation(
        validation_predictions, selection_metric=args.monitor_metric
    )
    selected_aggregation = aggregation_selection.selected_aggregation
    threshold_selection = select_operating_threshold_on_validation(
        validation_predictions.labels,
        validation_predictions.probabilities_by_aggregation[selected_aggregation],
        objective=args.threshold_objective,
    )
    validation_report = {
        "split": "validation",
        "model": "finetuned_resnet18_study_level",
        "best_checkpoint": str(checkpoint_path),
        "training_result": training_result.to_dict(),
        "aggregation_selection_on_validation": aggregation_selection.to_dict(),
        "operating_threshold_selected_on_validation": threshold_selection.to_dict(),
        "class_weighting": class_weight_decision.to_dict(),
        "selected_aggregation_report": _prediction_report(
            validation_predictions,
            selected_aggregation,
            threshold=threshold_selection.threshold,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "mean_at_0_5": metrics_at_default_and_selected_threshold(
            validation_predictions.labels,
            validation_predictions.probabilities_by_aggregation["mean"],
            0.5,
        )["threshold_0_5"],
        "max_at_0_5": metrics_at_default_and_selected_threshold(
            validation_predictions.labels,
            validation_predictions.probabilities_by_aggregation["max"],
            0.5,
        )["threshold_0_5"],
    }
    save_json(validation_report, output_dir / "validation_metrics.json")
    for method in ("mean", "max"):
        _write_rows(
            output_dir / f"validation_predictions_{method}.csv",
            validation_predictions.rows(method),
        )
    _write_rows(
        output_dir / "validation_predictions.csv",
        validation_predictions.rows(selected_aggregation),
    )
    selection_record = {
        "best_checkpoint": str(checkpoint_path),
        "checkpoint_monitor": {
            "metric": args.monitor_metric,
            "aggregation": args.monitor_aggregation,
        },
        "aggregation_selection": aggregation_selection.to_dict(),
        "threshold_selection": threshold_selection.to_dict(),
        "selection_data": "validation only",
        "test_used_for_selection": False,
    }
    save_json(selection_record, output_dir / "selection.json")
    run_config = {
        "seed": int(args.seed),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "model_provenance": model.provenance(),
        "class_weighting": class_weight_decision.to_dict(),
        "preprocessing": preprocessing_spec(args.image_size),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "train_studies": int(len(train_dataset)),
        "validation_studies": int(len(validation_dataset)),
    }
    save_json(run_config, output_dir / "run_config.json")
    outputs: dict[str, Path] = {
        "checkpoint": checkpoint_path,
        "history": output_dir / "training_history.json",
        "validation_metrics": output_dir / "validation_metrics.json",
        "selection": output_dir / "selection.json",
        "run_config": output_dir / "run_config.json",
    }

    # The test dataset is constructed only after all validation-driven choices
    # above have been persisted. Its labels/probabilities are never routed into
    # fit(), scheduler stepping, aggregation/threshold selection, or final
    # metric reporting; evaluate_phase1.py owns final test metrics.
    if args.write_test_predictions:
        test_dataset = MURAStudyDataset(
            args.manifest,
            dataset_root=args.dataset_root,
            split_manifest=args.split_manifest,
            split="test",
            transform=build_eval_transform(image_size=args.image_size),
        )
        _assert_disjoint_datasets(
            (("train", train_dataset), ("validation", validation_dataset), ("test", test_dataset))
        )
        test_loader = _make_loader(
            test_dataset,
            batch_size=args.batch_size,
            workers=args.workers,
            shuffle=False,
            seed=args.seed,
            pin_memory=device.type == "cuda",
        )
        test_predictions = trainer.predict(test_loader, aggregations=(selected_aggregation,))
        _write_rows(output_dir / "test_predictions.csv", test_predictions.rows(selected_aggregation))
        test_prediction_metadata = {
            "split": "test",
            "model": "finetuned_resnet18_study_level",
            "frozen_checkpoint": str(checkpoint_path),
            "frozen_aggregation_selected_on_validation": aggregation_selection.to_dict(),
            "frozen_operating_threshold_selected_on_validation": threshold_selection.to_dict(),
            "test_used_for_selection": False,
            "final_test_metrics_computed": False,
            "final_test_metrics_owner": "evaluate_phase1.py",
        }
        save_json(test_prediction_metadata, output_dir / "test_prediction_metadata.json")
        outputs["test_predictions"] = output_dir / "test_predictions.csv"
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        outputs = run(args)
    except PretrainedWeightsUnavailableError as exc:
        parser.error(str(exc))
    except (FileNotFoundError, ValueError, RuntimeError, OSError, FloatingPointError) as exc:
        parser.error(str(exc))
    print(f"Saved validation-best checkpoint: {outputs['checkpoint']}")
    print(f"Saved validation-only selection record: {outputs['selection']}")
    if "test_predictions" in outputs:
        print(
            "Saved sealed test predictions; final test metrics are deferred to "
            f"evaluate_phase1.py: {outputs['test_predictions']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
