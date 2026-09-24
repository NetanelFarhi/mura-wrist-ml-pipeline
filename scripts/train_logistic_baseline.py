#!/usr/bin/env python
"""Train the frozen-embedding Logistic Regression Phase 1 baseline.

The script accepts separate train and validation embedding artifacts, selects a
small C grid using validation AUROC/AUPRC only, then selects an operating
threshold from validation predictions only.  With ``--write-test-predictions``
it can generate a sealed test prediction CSV after those choices are written;
final test metrics and model comparison belong to ``evaluate_phase1.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.features.extract_embeddings import StudyEmbeddings, load_study_embeddings
from src.models.frozen_embedding_baseline import FrozenEmbeddingLogisticBaseline
from src.training.metrics import (
    bootstrap_auroc_ci,
    metrics_at_default_and_selected_threshold,
    select_operating_threshold_on_validation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-embeddings", type=Path, required=True)
    parser.add_argument("--val-embeddings", type=Path, required=True)
    parser.add_argument(
        "--test-embeddings",
        type=Path,
        default=None,
        help="Sealed test embeddings, required only with --write-test-predictions.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=(0.1, 1.0, 10.0),
        help="One to three validation candidates only (default: 0.1 1.0 10.0).",
    )
    parser.add_argument("--selection-metric", choices=("auroc", "auprc"), default="auroc")
    parser.add_argument(
        "--threshold-objective",
        choices=("youden_j", "f1", "balanced_accuracy"),
        default="youden_j",
    )
    parser.add_argument(
        "--class-weight",
        choices=("none", "balanced"),
        default="none",
        help="Optional explicit Logistic Regression class weighting; default is unweighted.",
    )
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument(
        "--refit-on-train-plus-val",
        action="store_true",
        help="After C/threshold are frozen, refit the selected C on train+val before optional test evaluation.",
    )
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_predictions(path: Path, artifact: StudyEmbeddings, probabilities: Sequence[float]) -> None:
    if len(probabilities) != len(artifact.labels):
        raise ValueError("Prediction count does not match embedding artifact rows.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("patient_id", "study_id", "label", "num_views", "abnormal_probability"),
        )
        writer.writeheader()
        for patient_id, study_id, label, num_views, probability in zip(
            artifact.patient_ids,
            artifact.study_ids,
            artifact.labels,
            artifact.num_views,
            probabilities,
        ):
            writer.writerow(
                {
                    "patient_id": str(patient_id),
                    "study_id": str(study_id),
                    "label": int(label),
                    "num_views": int(num_views),
                    "abnormal_probability": f"{float(probability):.10g}",
                }
            )


def _require_expected_split(artifact: StudyEmbeddings, expected: str, artifact_name: str) -> None:
    actual = artifact.metadata.get("final_split")
    if actual != expected:
        raise ValueError(
            f"{artifact_name} must declare metadata.final_split={expected!r}; found {actual!r}. "
            "Regenerate it with scripts/extract_embeddings.py from the correct patient-safe split."
        )


def _assert_disjoint(named_artifacts: Sequence[tuple[str, StudyEmbeddings]]) -> None:
    for index, (left_name, left) in enumerate(named_artifacts):
        for right_name, right in named_artifacts[index + 1 :]:
            study_overlap = set(map(str, left.study_ids)).intersection(map(str, right.study_ids))
            if study_overlap:
                raise ValueError(
                    f"Study leakage between {left_name} and {right_name}: {sorted(study_overlap)[:3]}"
                )
            left_patients = {str(value) for value in left.patient_ids if str(value)}
            right_patients = {str(value) for value in right.patient_ids if str(value)}
            patient_overlap = left_patients.intersection(right_patients)
            if patient_overlap:
                raise ValueError(
                    f"Patient leakage between {left_name} and {right_name}: {sorted(patient_overlap)[:3]}"
                )


def _report(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    selected_threshold: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "metrics": metrics_at_default_and_selected_threshold(
            labels, probabilities, selected_threshold
        ),
        "auroc_bootstrap_ci": bootstrap_auroc_ci(
            labels, probabilities, n_bootstrap=bootstrap_samples, seed=seed
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be at least 1.")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train = load_study_embeddings(args.train_embeddings)
    validation = load_study_embeddings(args.val_embeddings)
    _require_expected_split(train, "train", "--train-embeddings")
    _require_expected_split(validation, "val", "--val-embeddings")
    artifacts: list[tuple[str, StudyEmbeddings]] = [("train", train), ("validation", validation)]
    if args.write_test_predictions and args.test_embeddings is None:
        raise ValueError("--write-test-predictions requires --test-embeddings.")
    if args.test_embeddings is not None and not args.write_test_predictions:
        raise ValueError(
            "Pass --write-test-predictions to read sealed test embeddings. "
            "This CLI intentionally does not calculate final test metrics."
        )
    test: StudyEmbeddings | None = None
    if args.write_test_predictions:
        assert args.test_embeddings is not None
        test = load_study_embeddings(args.test_embeddings)
        _require_expected_split(test, "test", "--test-embeddings")
        artifacts.append(("test", test))
    _assert_disjoint(artifacts)

    baseline = FrozenEmbeddingLogisticBaseline(
        c_values=args.c_values,
        selection_metric=args.selection_metric,
        class_weight="balanced" if args.class_weight == "balanced" else None,
        random_seed=args.seed,
    ).fit(train.embeddings, train.labels, validation.embeddings, validation.labels)

    # These predictions belong to the train-only candidates and are the only
    # data used for C and threshold selection.
    validation_probabilities = baseline.predict_proba(validation.embeddings)
    threshold_selection = select_operating_threshold_on_validation(
        validation.labels,
        validation_probabilities,
        objective=args.threshold_objective,
    )
    validation_report = {
        "split": "validation",
        "model": "frozen_resnet18_embeddings_logistic_regression",
        "model_selection": baseline.selection_metadata(),
        "operating_threshold_selected_on_validation": threshold_selection.to_dict(),
        **_report(
            validation.labels,
            validation_probabilities,
            selected_threshold=threshold_selection.threshold,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    }
    _write_json(output_dir / "validation_metrics.json", validation_report)
    _write_predictions(output_dir / "validation_predictions.csv", validation, validation_probabilities)
    _write_json(
        output_dir / "selection.json",
        {
            "c_selection": baseline.selection_metadata(),
            "threshold_selection": threshold_selection.to_dict(),
            "selection_data": "validation only",
            "test_used_for_selection": False,
        },
    )

    if args.refit_on_train_plus_val:
        baseline.refit_on_train_and_validation_after_freeze(
            train.embeddings, train.labels, validation.embeddings, validation.labels
        )
    model_path = baseline.save(output_dir / "frozen_embedding_logistic.joblib")
    outputs: dict[str, Path] = {
        "model": model_path,
        "validation_metrics": output_dir / "validation_metrics.json",
        "selection": output_dir / "selection.json",
    }

    # Test data is intentionally read only here, after all validation choices
    # above are persisted.  It cannot enter fit(), C/threshold selection, or
    # final metric reporting; evaluate_phase1.py owns final test metrics.
    if test is not None:
        test_probabilities = baseline.predict_proba(test.embeddings)
        _write_predictions(output_dir / "test_predictions.csv", test, test_probabilities)
        test_prediction_metadata = {
            "split": "test",
            "model": "frozen_resnet18_embeddings_logistic_regression",
            "frozen_model_selection": baseline.selection_metadata(),
            "frozen_operating_threshold_selected_on_validation": threshold_selection.to_dict(),
            "test_used_for_selection": False,
            "final_test_metrics_computed": False,
            "final_test_metrics_owner": "evaluate_phase1.py",
        }
        _write_json(output_dir / "test_prediction_metadata.json", test_prediction_metadata)
        outputs["test_predictions"] = output_dir / "test_predictions.csv"
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        outputs = run(args)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    print(f"Saved baseline model: {outputs['model']}")
    print(f"Saved validation-only selection record: {outputs['selection']}")
    if "test_predictions" in outputs:
        print(
            "Saved sealed test predictions; final test metrics are deferred to "
            f"evaluate_phase1.py: {outputs['test_predictions']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
