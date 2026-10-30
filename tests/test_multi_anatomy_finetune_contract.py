"""Synthetic contract tests for all-anatomy fine-tuning inputs."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_multi_anatomy_finetuned import validate_development_inputs


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class MultiAnatomyFineTuneContractTests(unittest.TestCase):
    def make_valid_inputs(self, directory: Path) -> tuple[Path, Path]:
        manifest = directory / "xr_wrist_manifest.csv"
        split = directory / "xr_wrist_split.csv"
        fields = ["anatomy", "patient_id", "study_id", "image_path", "label", "source_split", "num_views_in_study"]
        rows = [
            {"anatomy": "XR_WRIST", "patient_id": "p1", "study_id": "XR_WRIST/p1/study1_negative", "image_path": "train/XR_WRIST/p1/study1_negative/image1.png", "label": "0", "source_split": "train", "num_views_in_study": "1"},
            {"anatomy": "XR_WRIST", "patient_id": "p2", "study_id": "XR_WRIST/p2/study1_positive", "image_path": "train/XR_WRIST/p2/study1_positive/image1.png", "label": "1", "source_split": "train", "num_views_in_study": "1"},
        ]
        write_csv(manifest, fields, rows)
        write_csv(split, ["patient_id", "study_id", "final_split"], [
            {"patient_id": "p1", "study_id": "XR_WRIST/p1/study1_negative", "final_split": "train"},
            {"patient_id": "p2", "study_id": "XR_WRIST/p2/study1_positive", "final_split": "val"},
        ])
        return manifest, split

    def test_train_only_patient_safe_inputs_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, split = self.make_valid_inputs(Path(temporary))
            validate_development_inputs(manifest, split, "XR_WRIST")

    def test_valid_path_is_rejected_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, split = self.make_valid_inputs(Path(temporary))
            contents = manifest.read_text(encoding="utf-8")
            manifest.write_text(contents.replace("train/XR_WRIST", "valid/XR_WRIST"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "valid/test"):
                validate_development_inputs(manifest, split, "XR_WRIST")

    def test_test_assignment_is_rejected_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, split = self.make_valid_inputs(Path(temporary))
            contents = split.read_text(encoding="utf-8")
            split.write_text(contents.replace(",val", ",test"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "never test"):
                validate_development_inputs(manifest, split, "XR_WRIST")

    def test_anatomy_mismatch_is_rejected_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, split = self.make_valid_inputs(Path(temporary))
            with self.assertRaisesRegex(ValueError, "Expected XR_HAND"):
                validate_development_inputs(manifest, split, "XR_HAND")


if __name__ == "__main__":
    unittest.main()
