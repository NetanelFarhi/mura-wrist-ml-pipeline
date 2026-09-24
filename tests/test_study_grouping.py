"""Tests that all XR_WRIST views remain grouped at study level."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collate import collate_studies  # noqa: E402
from src.data.study_dataset import MURAStudyDataset  # noqa: E402


def _row(patient: str, study: str, image: str, label: int, views: int, split: str) -> dict[str, object]:
    return {
        "patient_id": patient,
        "study_id": study,
        "image_path": image,
        "label": label,
        "source_split": "train",
        "num_views_in_study": views,
        "final_split": split,
    }


class StudyGroupingTests(unittest.TestCase):
    def test_dataset_filters_final_split_without_separating_views(self) -> None:
        rows = [
            _row("patient1", "patient1/study1_negative", "a/view2.png", 0, 2, "train"),
            _row("patient1", "patient1/study1_negative", "a/view1.png", 0, 2, "train"),
            _row("patient2", "patient2/study1_positive", "b/view1.png", 1, 1, "val"),
        ]
        train_dataset = MURAStudyDataset(rows, split="train")
        val_dataset = MURAStudyDataset(rows, split="val")

        self.assertEqual(len(train_dataset), 1)
        self.assertEqual(len(val_dataset), 1)
        train_study = train_dataset.studies[0]
        self.assertEqual(train_study.num_views, 2)
        self.assertEqual(train_study.image_paths, ("a/view1.png", "a/view2.png"))
        self.assertEqual(train_study.label, 0)

    def test_ragged_collation_emits_mask_for_each_real_view(self) -> None:
        batch = [
            {
                "images": ["view_a", "view_b"],
                "label": 0,
                "patient_id": "patient1",
                "study_id": "patient1/study1_negative",
                "image_paths": ["a", "b"],
            },
            {
                "images": ["view_c"],
                "label": 1,
                "patient_id": "patient2",
                "study_id": "patient2/study1_positive",
                "image_paths": ["c"],
            },
        ]
        collated = collate_studies(batch)
        self.assertEqual(collated["num_views"], [2, 1])
        self.assertEqual(collated["view_mask"], [[True, True], [True, False]])
        self.assertEqual(collated["study_ids"], ["patient1/study1_negative", "patient2/study1_positive"])


if __name__ == "__main__":
    unittest.main()
