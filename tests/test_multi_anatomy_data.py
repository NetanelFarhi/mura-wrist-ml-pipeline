"""Synthetic safety tests for the all-anatomy MURA development track."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.multi_anatomy import (
    MultiAnatomyRecord,
    build_train_manifest,
    global_patient_split,
    parse_train_image_path,
)


def make_record(anatomy: str, patient: str, study: str, label: int) -> MultiAnatomyRecord:
    return MultiAnatomyRecord(
        anatomy=anatomy,
        patient_id=patient,
        study_id=f"{anatomy}/{patient}/{study}",
        image_path=f"train/{anatomy}/{patient}/{study}/image1.png",
        label=label,
        source_split="train",
        num_views_in_study=1,
    )


class MultiAnatomyDataTests(unittest.TestCase):
    def test_anatomy_is_part_of_study_identity(self) -> None:
        wrist = parse_train_image_path(
            "MURA-v1.1/train/XR_WRIST/patient00001/study1_positive/image1.png"
        )
        shoulder = parse_train_image_path(
            "MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/image1.png"
        )
        self.assertNotEqual(wrist.study_id, shoulder.study_id)
        self.assertEqual(wrist.study_id, "XR_WRIST/patient00001/study1_positive")
        self.assertEqual(shoulder.study_id, "XR_SHOULDER/patient00001/study1_positive")

    def test_official_valid_path_is_rejected_before_manifest_creation(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden"):
            parse_train_image_path(
                "MURA-v1.1/valid/XR_WRIST/patient00001/study1_positive/image1.png"
            )

    def test_global_split_keeps_one_patient_together_across_anatomies(self) -> None:
        records = [
            make_record("XR_WRIST", "patient01", "study1_positive", 1),
            make_record("XR_SHOULDER", "patient01", "study1_negative", 0),
            make_record("XR_WRIST", "patient02", "study1_negative", 0),
            make_record("XR_SHOULDER", "patient02", "study1_positive", 1),
            make_record("XR_WRIST", "patient03", "study1_positive", 1),
            make_record("XR_SHOULDER", "patient03", "study1_negative", 0),
            make_record("XR_WRIST", "patient04", "study1_negative", 0),
            make_record("XR_SHOULDER", "patient04", "study1_positive", 1),
        ]
        split_rows = global_patient_split(records, folds=2, seed=19)
        assignments: dict[str, set[str]] = {}
        for row in split_rows:
            assignments.setdefault(row.patient_id, set()).add(row.final_split)
        self.assertTrue(all(len(value) == 1 for value in assignments.values()))

    def test_train_manifest_rejects_valid_rows_inside_source_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "train_image_paths.csv"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["MURA-v1.1/train/XR_WRIST/patient00001/study1_positive/image1.png"])
                writer.writerow(["MURA-v1.1/valid/XR_WRIST/patient99999/study1_positive/image1.png"])
            with self.assertRaisesRegex(ValueError, "valid"):
                build_train_manifest(source)


if __name__ == "__main__":
    unittest.main()
