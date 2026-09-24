"""Synthetic, no-MURA tests for Phase-1 split and evaluation safeguards."""

from __future__ import annotations

import json
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.leakage_checks import (
    LeakageError,
    validate_evaluation_provenance,
    validate_split_integrity,
    validate_threshold_provenance,
)
from src.evaluation.model_eval import (
    evaluate_test_with_frozen_validation_threshold,
    evaluate_test_with_validation_threshold,
    select_validation_threshold,
)


def synthetic_manifest() -> pd.DataFrame:
    """Four patients, one study each, with multi-image rows where relevant."""

    return pd.DataFrame(
        [
            ("patient-1", "study-1", "p1/s1/image1.png", 0, "train"),
            ("patient-1", "study-1", "p1/s1/image2.png", 0, "train"),
            ("patient-2", "study-2", "p2/s2/image1.png", 1, "train"),
            ("patient-3", "study-3", "p3/s3/image1.png", 0, "validation"),
            ("patient-4", "study-4", "p4/s4/image1.png", 1, "test"),
            ("patient-4", "study-4", "p4/s4/image2.png", 1, "test"),
        ],
        columns=["patient_id", "study_id", "image_path", "label", "final_split"],
    )


class TestNoTestLeakage(unittest.TestCase):
    def test_patient_safe_manifest_passes_all_integrity_checks(self) -> None:
        report = validate_split_integrity(synthetic_manifest())
        self.assertTrue(report.valid)
        self.assertEqual(report.split_counts, {"test": 2, "train": 3, "validation": 1})

    def test_patient_overlap_between_train_and_validation_is_rejected(self) -> None:
        manifest = synthetic_manifest()
        manifest.loc[manifest["study_id"] == "study-3", "patient_id"] = "patient-1"
        with self.assertRaisesRegex(LeakageError, "patient_overlap_across_splits"):
            validate_split_integrity(manifest)

    def test_study_images_cannot_be_split_between_train_and_test(self) -> None:
        manifest = synthetic_manifest()
        manifest.loc[
            (manifest["study_id"] == "study-1") & (manifest["image_path"].str.endswith("image2.png")),
            "final_split",
        ] = "test"
        with self.assertRaisesRegex(LeakageError, "study_overlap_across_splits"):
            validate_split_integrity(manifest)

    def test_study_labels_must_be_internally_consistent(self) -> None:
        manifest = synthetic_manifest()
        manifest.loc[
            (manifest["study_id"] == "study-1") & (manifest["image_path"].str.endswith("image2.png")),
            "label",
        ] = 1
        with self.assertRaisesRegex(LeakageError, "inconsistent_labels_within_study"):
            validate_split_integrity(manifest)

    def test_test_split_is_rejected_for_model_fit_or_threshold_selection(self) -> None:
        unsafe_fit = {
            "model_fit_splits": ["train", "test"],
            "preprocessing_fit_splits": ["train"],
            "threshold_selection_splits": ["validation"],
        }
        with self.assertRaisesRegex(LeakageError, "test_data_used_for_fitting_or_selection"):
            validate_evaluation_provenance(unsafe_fit)

        unsafe_threshold = {"threshold_selection_splits": ["test"]}
        with self.assertRaisesRegex(LeakageError, "test_data_used_for_threshold_selection"):
            validate_threshold_provenance(unsafe_threshold)

    def test_valid_recorded_provenance_passes(self) -> None:
        provenance = {
            "model_fit_splits": ["train"],
            "preprocessing_fit_splits": ["train"],
            "hyperparameter_selection_splits": ["validation"],
            "aggregation_selection_splits": ["validation"],
            "threshold_selection_splits": ["validation"],
        }
        report = validate_evaluation_provenance(provenance)
        self.assertTrue(report["fitting"]["valid"])
        self.assertTrue(report["threshold"]["valid"])

    def test_threshold_helper_refuses_a_test_source(self) -> None:
        predictions = pd.DataFrame(
            {
                "study_id": ["s1", "s2", "s3", "s4"],
                "label": [0, 0, 1, 1],
                "probability": [0.1, 0.4, 0.6, 0.9],
                "num_views": [1, 2, 1, 2],
                "final_split": ["test"] * 4,
            }
        )
        with self.assertRaisesRegex(ValueError, "requires 'validation' data"):
            select_validation_threshold(predictions, source_split="test")

    def test_test_metrics_use_a_validation_selected_threshold(self) -> None:
        validation = pd.DataFrame(
            {
                "study_id": ["v1", "v2", "v3", "v4"],
                "label": [0, 0, 1, 1],
                "probability": [0.05, 0.30, 0.70, 0.95],
                "num_views": [1, 2, 1, 2],
                "final_split": ["validation"] * 4,
            }
        )
        test = pd.DataFrame(
            {
                "study_id": ["t1", "t2", "t3", "t4"],
                "label": [0, 0, 1, 1],
                "probability": [0.20, 0.40, 0.65, 0.90],
                "num_views": [1, 1, 2, 2],
                "final_split": ["test"] * 4,
            }
        )
        result = evaluate_test_with_validation_threshold(
            validation, test, bootstrap_samples=20, random_state=7
        )
        self.assertEqual(result["prediction_unit"], "study")
        self.assertEqual(
            result["validation_selected_threshold"]["selection_source"], "validation"
        )
        self.assertIn("threshold_0_5", result)
        self.assertIn("validation_selected", result)
        self.assertIn("false_negative_rate", result["validation_selected"])

    def test_cli_writes_reports_from_synthetic_predictions_without_mura(self) -> None:
        validation = pd.DataFrame(
            {
                "study_id": ["v1", "v2", "v3", "v4"],
                "label": [0, 0, 1, 1],
                "abnormal_probability": [0.10, 0.30, 0.70, 0.90],
                "num_views": [1, 2, 1, 2],
                "final_split": ["val"] * 4,
            }
        )
        test = pd.DataFrame(
            {
                "study_id": ["t1", "t2", "t3", "t4"],
                "label": [0, 0, 1, 1],
                "abnormal_probability": [0.20, 0.40, 0.60, 0.80],
                "num_views": [1, 1, 2, 2],
                "final_split": ["test"] * 4,
            }
        )
        with tempfile.TemporaryDirectory() as temp_directory:
            temp = Path(temp_directory)
            validation_path = temp / "validation.csv"
            test_path = temp / "test.csv"
            metrics_dir = temp / "metrics"
            errors_dir = temp / "errors"
            baseline_report = temp / "baseline_results.md"
            error_report = temp / "error_analysis.md"
            selection_record = temp / "selection.json"
            manifest_path = temp / "wrist_manifest.csv"
            split_manifest_path = temp / "split_manifest.csv"
            validation.to_csv(validation_path, index=False)
            test.to_csv(test_path, index=False)
            selection_record.write_text(
                json.dumps(
                    {
                        "selection_data": "validation only",
                        "test_used_for_selection": False,
                        "threshold_selection": {
                            "threshold": 0.5,
                            "objective": "youden_j",
                            "objective_value": 1.0,
                            "candidates_evaluated": 5,
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest_rows = pd.DataFrame(
                [
                    ("p-train", "s-train", "p-train/s-train.png", 0, "train", 1),
                    ("p-v1", "v1", "p-v1/v1.png", 0, "train", 1),
                    ("p-v2", "v2", "p-v2/v2.png", 0, "train", 1),
                    ("p-v3", "v3", "p-v3/v3.png", 1, "train", 1),
                    ("p-v4", "v4", "p-v4/v4.png", 1, "train", 1),
                    ("p-t1", "t1", "p-t1/t1.png", 0, "valid", 1),
                    ("p-t2", "t2", "p-t2/t2.png", 0, "valid", 1),
                    ("p-t3", "t3", "p-t3/t3.png", 1, "valid", 1),
                    ("p-t4", "t4", "p-t4/t4.png", 1, "valid", 1),
                ],
                columns=[
                    "patient_id",
                    "study_id",
                    "image_path",
                    "label",
                    "source_split",
                    "num_views_in_study",
                ],
            )
            split_rows = pd.DataFrame(
                [
                    ("p-train", "s-train", "train"),
                    ("p-v1", "v1", "val"),
                    ("p-v2", "v2", "val"),
                    ("p-v3", "v3", "val"),
                    ("p-v4", "v4", "val"),
                    ("p-t1", "t1", "test"),
                    ("p-t2", "t2", "test"),
                    ("p-t3", "t3", "test"),
                    ("p-t4", "t4", "test"),
                ],
                columns=["patient_id", "study_id", "final_split"],
            )
            manifest_rows.to_csv(manifest_path, index=False)
            split_rows.to_csv(split_manifest_path, index=False)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "evaluate_phase1.py"),
                    "--model-name",
                    "Synthetic baseline",
                    "--validation-predictions",
                    str(validation_path),
                    "--test-predictions",
                    str(test_path),
                    "--selection-record",
                    str(selection_record),
                    "--manifest",
                    str(manifest_path),
                    "--split-manifest",
                    str(split_manifest_path),
                    "--output-dir",
                    str(metrics_dir),
                    "--error-output-dir",
                    str(errors_dir),
                    "--baseline-report",
                    str(baseline_report),
                    "--error-report",
                    str(error_report),
                    "--no-bootstrap",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((metrics_dir / "model_comparison.csv").is_file())
            self.assertTrue(baseline_report.is_file())
            self.assertTrue(error_report.is_file())
            self.assertIn("study-level", baseline_report.read_text(encoding="utf-8"))
            self.assertIn("does not establish a clinical cause", error_report.read_text(encoding="utf-8"))
            evaluation = json.loads(
                (metrics_dir / "Synthetic_baseline_evaluation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                evaluation["validation_selected_threshold"]["selection_mode"],
                "frozen_validation_record",
            )
            metadata = json.loads(
                (metrics_dir / "evaluation_run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["models"]["Synthetic baseline"]["probability_column"],
                "abnormal_probability",
            )

    def test_frozen_threshold_evaluation_does_not_reselect_on_test(self) -> None:
        test = pd.DataFrame(
            {
                "study_id": ["t1", "t2", "t3", "t4"],
                "label": [0, 0, 1, 1],
                "abnormal_probability": [0.20, 0.40, 0.60, 0.80],
                "num_views": [1, 1, 2, 2],
                "final_split": ["test"] * 4,
            }
        )
        result = evaluate_test_with_frozen_validation_threshold(
            test,
            frozen_threshold=0.73,
            frozen_selection_metadata={"selection_record": "validation-only.json"},
            probability_column="abnormal_probability",
            bootstrap_samples=20,
            random_state=7,
        )
        self.assertEqual(result["validation_selected"]["threshold"], 0.73)
        self.assertEqual(
            result["validation_selected_threshold"]["selection_mode"],
            "frozen_validation_record",
        )


if __name__ == "__main__":
    unittest.main()
