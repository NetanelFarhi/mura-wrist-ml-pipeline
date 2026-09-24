"""Evaluate Phase-1 study predictions without loading MURA images.

Examples
--------
Evaluate one model from two study-level CSV files::

    python scripts/evaluate_phase1.py \
      --model-name "Frozen ResNet-18 + Logistic Regression" \
      --validation-predictions reports/metrics/logistic_validation_predictions.csv \
      --test-predictions reports/metrics/logistic_test_predictions.csv \
      --selection-record models/frozen_logistic/selection.json \
      --manifest data/manifests/wrist_manifest.csv \
      --split-manifest data/manifests/split_manifest.csv

For the two-model comparison, use a JSON configuration::

    {
      "Frozen ResNet-18 + Logistic Regression": {
        "validation_predictions": "reports/metrics/logistic_validation_predictions.csv",
        "test_predictions": "reports/metrics/logistic_test_predictions.csv",
        "selection_record": "models/frozen_logistic/selection.json",
        "probability_column": "abnormal_probability"
      },
      "Fine-tuned ResNet-18": {
        "validation_predictions": "reports/metrics/resnet_validation_predictions.csv",
        "test_predictions": "reports/metrics/resnet_test_predictions.csv",
        "selection_record": "models/resnet18/selection.json",
        "probability_column": "abnormal_probability"
      }
    }

Each prediction CSV must have one row per study and columns ``study_id``,
``label``, ``num_views``, and either ``abnormal_probability`` (the current
Model A/B output contract) or ``probability``. ``num_views`` may instead be
derived from ``--manifest``. Final evaluation consumes the validation threshold
already frozen in each model's ``selection.json``; it does not select again.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.error_analysis import write_error_analysis_reports
from src.evaluation.leakage_checks import audit_split_integrity, validate_evaluation_provenance
from src.evaluation.model_eval import (
    build_model_comparison_table,
    evaluate_test_with_frozen_validation_threshold,
    evaluate_test_with_validation_threshold,
    persist_evaluation_results,
)
from src.data.split_patients import join_manifest_with_splits


@dataclass(frozen=True)
class ModelSpec:
    name: str
    validation_predictions: Path
    test_predictions: Path
    selection_record: Path | None = None
    probability_column: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study-level Phase-1 evaluation with sealed-test safeguards."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--evaluation-config",
        type=Path,
        help="JSON mapping model names to validation_predictions and test_predictions paths.",
    )
    input_group.add_argument(
        "--model-name",
        help="Name for a single model (requires --validation-predictions and --test-predictions).",
    )
    parser.add_argument("--validation-predictions", type=Path)
    parser.add_argument("--test-predictions", type=Path)
    parser.add_argument(
        "--selection-record",
        type=Path,
        help=(
            "Frozen validation-only selection JSON for a single model. Required unless "
            "--allow-validation-threshold-reselection is explicitly used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "metrics",
        help="Directory for JSON/CSV metrics (default: reports/metrics).",
    )
    parser.add_argument(
        "--error-output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "errors",
        help="Directory for representative error-analysis indices (default: reports/errors).",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "baseline_results.md",
        help="Markdown metrics report to write.",
    )
    parser.add_argument(
        "--error-report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "error_analysis.md",
        help="Combined descriptive error-analysis report to write.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Optional image-level wrist manifest for split integrity checks and "
            "deriving num_views when the prediction CSV lacks it."
        ),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help=(
            "Patient-safe split_manifest.csv. Required with a six-column wrist manifest "
            "so it can be joined and audited before final evaluation."
        ),
    )
    parser.add_argument(
        "--provenance-json",
        type=Path,
        help=(
            "Optional recorded split provenance. When supplied, model/preprocessing "
            "and threshold sources are validated and saved."
        ),
    )
    parser.add_argument("--study-id-column", default="study_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--probability-column",
        default=None,
        help=(
            "Override the prediction-probability column. By default the evaluator "
            "detects probability or abnormal_probability from each model's CSVs."
        ),
    )
    parser.add_argument("--num-views-column", default="num_views")
    parser.add_argument(
        "--split-column",
        default="final_split",
        help="Optional split-label column in prediction CSVs/manifests.",
    )
    parser.add_argument(
        "--threshold-criterion",
        choices=["youden", "f1", "balanced_accuracy", "sensitivity_at_specificity"],
        default="youden",
    )
    parser.add_argument("--min-specificity", type=float)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-confidence-level", type=float, default=0.95)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip AUROC bootstrap CI (useful for a very fast smoke test).",
    )
    parser.add_argument("--high-confidence", type=float, default=0.80)
    parser.add_argument("--low-confidence-max", type=float, default=0.60)
    parser.add_argument("--max-error-examples", type=int, default=25)
    parser.add_argument(
        "--allow-validation-threshold-reselection",
        action="store_true",
        help=(
            "Permit validation threshold reselection only for a synthetic/smoke run. "
            "Normal Phase-1 final evaluation must use each model's frozen selection record."
        ),
    )
    return parser.parse_args()


def _resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    if args.model_name is not None:
        if args.validation_predictions is None or args.test_predictions is None:
            raise ValueError(
                "--model-name requires both --validation-predictions and --test-predictions"
            )
        return [
            ModelSpec(
                name=args.model_name,
                validation_predictions=_resolve_path(
                    args.validation_predictions, base_dir=Path.cwd()
                ),
                test_predictions=_resolve_path(args.test_predictions, base_dir=Path.cwd()),
                selection_record=(
                    _resolve_path(args.selection_record, base_dir=Path.cwd())
                    if args.selection_record is not None
                    else None
                ),
                probability_column=args.probability_column,
            )
        ]

    config_path = _resolve_path(args.evaluation_config, base_dir=Path.cwd())
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if isinstance(config, Mapping) and "models" in config:
        config = config["models"]
    entries: list[tuple[str, Mapping[str, object]]] = []
    if isinstance(config, Mapping):
        entries = [(str(name), spec) for name, spec in config.items()]
    elif isinstance(config, list):
        for item in config:
            if not isinstance(item, Mapping) or "name" not in item:
                raise ValueError("Each model list entry needs a name field")
            entries.append((str(item["name"]), item))
    else:
        raise ValueError("Evaluation config must be a model mapping or a list of model objects")

    specs: list[ModelSpec] = []
    for name, spec in entries:
        if not isinstance(spec, Mapping):
            raise ValueError(f"Model specification for {name!r} must be an object")
        try:
            validation = spec["validation_predictions"]
            test = spec["test_predictions"]
        except KeyError as error:
            raise ValueError(
                f"Model {name!r} needs validation_predictions and test_predictions"
            ) from error
        specs.append(
            ModelSpec(
                name=name,
                validation_predictions=_resolve_path(validation, base_dir=config_path.parent),
                test_predictions=_resolve_path(test, base_dir=config_path.parent),
                selection_record=(
                    _resolve_path(spec["selection_record"], base_dir=config_path.parent)
                    if spec.get("selection_record") is not None
                    else None
                ),
                probability_column=(
                    str(spec["probability_column"])
                    if spec.get("probability_column") is not None
                    else args.probability_column
                ),
            )
        )
    if not specs:
        raise ValueError("Evaluation config contains no model specifications")
    return specs


def _load_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    return pd.read_csv(path)


def _resolve_probability_column(
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    *,
    configured_column: str | None,
) -> str:
    """Resolve a shared emitted probability column without rewriting CSVs."""

    if configured_column is not None:
        candidates = [configured_column]
    else:
        # ``abnormal_probability`` is emitted by both current training scripts;
        # ``probability`` remains supported for interoperable external exports.
        candidates = ["abnormal_probability", "probability"]
    for candidate in candidates:
        if candidate in validation_predictions.columns and candidate in test_predictions.columns:
            return candidate
    required = configured_column or "one of abnormal_probability, probability"
    raise ValueError(
        f"Validation and test prediction CSVs must both contain {required!r}; "
        f"validation has {list(validation_predictions.columns)!r}, "
        f"test has {list(test_predictions.columns)!r}."
    )


def _load_frozen_validation_selection(path: Path) -> dict[str, Any]:
    """Read the validation-only threshold saved by a Model A/B training run."""

    if not path.is_file():
        raise FileNotFoundError(f"Selection record not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, Mapping):
        raise ValueError(f"Selection record must be a JSON object: {path}")
    if str(record.get("selection_data", "")).strip().lower() != "validation only":
        raise ValueError(
            f"Selection record must state selection_data='validation only': {path}"
        )
    if record.get("test_used_for_selection") is not False:
        raise ValueError(
            f"Selection record must explicitly state test_used_for_selection=false: {path}"
        )
    selection = record.get("threshold_selection")
    if not isinstance(selection, Mapping):
        # This alternate field is present in the validation report emitted by
        # both trainers and is useful when a caller persists that report only.
        selection = record.get("operating_threshold_selected_on_validation")
    if not isinstance(selection, Mapping) or "threshold" not in selection:
        raise ValueError(f"Selection record has no frozen threshold_selection.threshold: {path}")
    try:
        threshold = float(selection["threshold"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Frozen threshold is not numeric in {path}") from error
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Frozen threshold must be in [0, 1], found {threshold} in {path}")
    return {
        "threshold": threshold,
        "selection_record": str(path),
        "selection_data": "validation only",
        "test_used_for_selection": False,
        **{
            key: value
            for key, value in selection.items()
            if key in {"objective", "objective_value", "candidates_evaluated", "tie_break"}
        },
    }


def _load_and_audit_manifest(args: argparse.Namespace) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    """Join the exact source manifest to final splits before leakage auditing."""

    if args.manifest is None:
        return None, None
    manifest_path = _resolve_path(args.manifest, base_dir=Path.cwd())
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    if args.split_manifest is not None:
        split_path = _resolve_path(args.split_manifest, base_dir=Path.cwd())
        if not split_path.is_file():
            raise FileNotFoundError(f"Split manifest not found: {split_path}")
        manifest = pd.DataFrame(join_manifest_with_splits(manifest_path, split_path))
    elif args.split_column not in manifest.columns:
        raise ValueError(
            f"Manifest {manifest_path} has no {args.split_column!r}. Pass --split-manifest "
            "to join the patient-safe assignments before evaluation."
        )
    integrity = audit_split_integrity(
        manifest,
        split_column=args.split_column,
        patient_column="patient_id",
        study_column=args.study_id_column,
        label_column=args.label_column,
        image_column="image_path" if "image_path" in manifest.columns else None,
    )
    integrity.raise_if_invalid()
    return manifest, integrity.as_dict()


def _ensure_num_views(
    predictions: pd.DataFrame,
    *,
    manifest: pd.DataFrame | None,
    study_id_column: str,
    num_views_column: str,
) -> pd.DataFrame:
    """Use recorded study view counts, never an inferred image-level prediction."""

    result = predictions.copy()
    if num_views_column in result.columns:
        return result
    # The manifest contract calls this num_views_in_study. Accept it in a
    # study-level export without silently fabricating view counts.
    if "num_views_in_study" in result.columns:
        return result.rename(columns={"num_views_in_study": num_views_column})
    if manifest is None:
        raise ValueError(
            f"Prediction CSV lacks {num_views_column!r}; provide it or pass --manifest "
            "so view counts can be derived from actual image rows."
        )
    if study_id_column not in manifest.columns:
        raise ValueError(f"Manifest lacks {study_id_column!r} needed to derive num_views")
    counts = manifest.groupby(study_id_column, dropna=False).size()
    result[num_views_column] = result[study_id_column].map(counts)
    if result[num_views_column].isna().any():
        missing = result.loc[result[num_views_column].isna(), study_id_column].head(10)
        raise ValueError(
            "Prediction studies missing from manifest: "
            + ", ".join(str(value) for value in missing)
        )
    return result


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    return "\n".join(lines)


def _write_baseline_report(
    path: Path,
    comparison: pd.DataFrame,
    evaluations: Mapping[str, Mapping[str, Any]],
    *,
    provenance_status: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    at_half = build_model_comparison_table(evaluations, operating_point="threshold_0_5")
    lines = [
        "# Phase 1 baseline evaluation results",
        "",
        "All values are study-level results. Operating thresholds in the primary "
        "table were selected from validation predictions only; the final test data "
        "were used only to calculate the reported metrics.",
        "",
        "## Validation-selected operating threshold",
        "",
        _markdown_table(comparison),
        "",
        "## Fixed threshold = 0.5",
        "",
        _markdown_table(at_half),
        "",
        "## AUROC bootstrap intervals",
        "",
        "| Model | AUROC | 95% CI |",
        "|---|---:|---:|",
    ]
    for name, evaluation in evaluations.items():
        selected = evaluation["validation_selected"]
        ci = selected.get("auroc_bootstrap_ci")
        if isinstance(ci, Mapping):
            interval = f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
        else:
            interval = "Not computed"
        lines.append(f"| {name} | {selected['auroc']:.4f} | {interval} |")
    lines.extend(
        [
            "",
            "## Integrity/provenance evidence",
            "",
            provenance_status,
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_combined_error_report(
    path: Path,
    model_error_reports: Mapping[str, Mapping[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 1 descriptive error analysis",
        "",
        "This report indexes observable prediction outcomes, probabilities, and view "
        "counts. It does not establish a clinical cause for any error.",
        "",
    ]
    for name, artifacts in model_error_reports.items():
        report_path = Path(artifacts["markdown"])
        lines.extend([f"## {name}", ""])
        if report_path.is_file():
            body = report_path.read_text(encoding="utf-8")
            # The model-specific heading would be redundant in the combined file.
            body = body.replace("# Phase 1 error-analysis index\n\n", "", 1)
            lines.extend([body.rstrip(), ""])
        else:
            lines.extend(["No error-analysis index was written.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    try:
        specs = _model_specs(args)
        manifest, integrity_report = _load_and_audit_manifest(args)
        provenance_status = (
            "No global provenance JSON was supplied. Frozen model-specific validation "
            "threshold records were checked, but model/preprocessing fitting provenance "
            "was not independently verified in this run."
        )
        run_metadata: dict[str, Any] = {"models": {}, "integrity": None, "provenance": None}

        if integrity_report is not None:
            run_metadata["integrity"] = integrity_report
            provenance_status = "Image-manifest split integrity check: passed."

        if args.provenance_json is not None:
            provenance_path = _resolve_path(args.provenance_json, base_dir=Path.cwd())
            with provenance_path.open("r", encoding="utf-8") as handle:
                provenance = json.load(handle)
            run_metadata["provenance"] = validate_evaluation_provenance(provenance)
            provenance_status += (
                " Recorded model/preprocessing and threshold provenance check: passed."
            )

        evaluations: dict[str, Mapping[str, Any]] = {}
        error_artifacts: dict[str, Mapping[str, str]] = {}
        bootstrap_samples = 0 if args.no_bootstrap else args.bootstrap_samples
        for spec in specs:
            validation = _ensure_num_views(
                _load_predictions(spec.validation_predictions),
                manifest=manifest,
                study_id_column=args.study_id_column,
                num_views_column=args.num_views_column,
            )
            test = _ensure_num_views(
                _load_predictions(spec.test_predictions),
                manifest=manifest,
                study_id_column=args.study_id_column,
                num_views_column=args.num_views_column,
            )
            probability_column = _resolve_probability_column(
                validation,
                test,
                configured_column=spec.probability_column,
            )
            frozen_selection: dict[str, Any] | None = None
            if spec.selection_record is not None:
                frozen_selection = _load_frozen_validation_selection(spec.selection_record)
                evaluation = evaluate_test_with_frozen_validation_threshold(
                    test,
                    frozen_threshold=float(frozen_selection["threshold"]),
                    frozen_selection_metadata=frozen_selection,
                    split_column=args.split_column,
                    study_id_column=args.study_id_column,
                    label_column=args.label_column,
                    probability_column=probability_column,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_confidence_level=args.bootstrap_confidence_level,
                    random_state=args.random_state,
                )
            elif args.allow_validation_threshold_reselection:
                evaluation = evaluate_test_with_validation_threshold(
                    validation,
                    test,
                    threshold_criterion=args.threshold_criterion,
                    min_specificity=args.min_specificity,
                    split_column=args.split_column,
                    study_id_column=args.study_id_column,
                    label_column=args.label_column,
                    probability_column=probability_column,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_confidence_level=args.bootstrap_confidence_level,
                    random_state=args.random_state,
                )
                evaluation["validation_selected_threshold"]["selection_mode"] = (
                    "explicit_validation_reselection"
                )
            else:
                raise ValueError(
                    f"Model {spec.name!r} has no selection_record. Final evaluation requires "
                    "the frozen validation threshold emitted by training; use "
                    "--allow-validation-threshold-reselection only for a synthetic smoke run."
                )
            evaluations[spec.name] = evaluation
            error_artifacts[spec.name] = write_error_analysis_reports(
                test,
                args.error_output_dir,
                threshold=float(evaluation["validation_selected"]["threshold"]),
                high_confidence=args.high_confidence,
                low_confidence_max=args.low_confidence_max,
                max_examples_per_category=args.max_error_examples,
                study_id_column=args.study_id_column,
                label_column=args.label_column,
                probability_column=probability_column,
                num_views_column=args.num_views_column,
                filename_prefix=spec.name,
            )
            run_metadata["models"][spec.name] = {
                "validation_predictions": str(spec.validation_predictions),
                "test_predictions": str(spec.test_predictions),
                "probability_column": probability_column,
                "selection_record": str(spec.selection_record) if spec.selection_record else None,
                "frozen_selection": frozen_selection,
                "error_artifacts": error_artifacts[spec.name],
            }

        comparison = build_model_comparison_table(evaluations)
        for name, evaluation in evaluations.items():
            persist_evaluation_results(
                name,
                evaluation,
                args.output_dir,
                comparison_table=comparison,
            )
        run_metadata["comparison_csv"] = str(args.output_dir / "model_comparison.csv")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run_metadata_path = args.output_dir / "evaluation_run_metadata.json"
        run_metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")

        _write_baseline_report(
            args.baseline_report,
            comparison,
            evaluations,
            provenance_status=provenance_status,
        )
        _write_combined_error_report(args.error_report, error_artifacts)
        print(f"Wrote study-level metrics: {args.output_dir}")
        print(f"Wrote comparison: {args.output_dir / 'model_comparison.csv'}")
        print(f"Wrote baseline report: {args.baseline_report}")
        print(f"Wrote error-analysis report: {args.error_report}")
        return 0
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
