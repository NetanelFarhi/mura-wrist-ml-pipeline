"""Safety checks for the bounded Phase 2C train-only workflow."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_phase2c import assert_train_only_inputs  # noqa: E402
from src.evaluation.paired_bootstrap import paired_patient_bootstrap  # noqa: E402


class Phase2CIntegrityTests(unittest.TestCase):
    def _write_manifests(self, root: Path, source_split: str, image_path: str) -> tuple[Path, Path]:
        manifest = root / "manifest.csv"
        split = root / "split.csv"
        pd.DataFrame(
            [
                {"patient_id": "p1", "study_id": "s1", "image_path": image_path, "label": 0, "source_split": source_split},
                {"patient_id": "p2", "study_id": "s2", "image_path": "train/XR_WRIST/p2/study/image.png", "label": 1, "source_split": source_split},
            ]
        ).to_csv(manifest, index=False)
        pd.DataFrame(
            [
                {"patient_id": "p1", "study_id": "s1", "final_split": "train"},
                {"patient_id": "p2", "study_id": "s2", "final_split": "val"},
            ]
        ).to_csv(split, index=False)
        return manifest, split

    def test_train_only_manifest_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, split = self._write_manifests(Path(directory), "train", "train/XR_WRIST/p1/study/image.png")
            summary = assert_train_only_inputs(manifest, split)
        self.assertEqual(summary["studies"], 2)
        self.assertEqual(summary["validation_studies"], 1)

    def test_non_train_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, split = self._write_manifests(Path(directory), "valid", "valid/XR_WRIST/p1/study/image.png")
            with self.assertRaises(ValueError):
                assert_train_only_inputs(manifest, split)

    def test_paired_bootstrap_reports_candidate_minus_reference(self) -> None:
        labels = np.array([0, 1, 0, 1, 0, 1])
        patients = np.array(["p1", "p1", "p2", "p3", "p4", "p5"])
        frozen = np.array([0.10, 0.65, 0.40, 0.55, 0.35, 0.70])
        candidate = np.array([0.05, 0.80, 0.20, 0.75, 0.15, 0.90])
        result = paired_patient_bootstrap(labels, frozen, candidate, patients, n_bootstrap=40, seed=4)
        self.assertEqual(set(result["metric"]), {"AUROC", "AUPRC", "Specificity@95%"})
        self.assertTrue((result["n_bootstrap_valid"] > 0).all())


if __name__ == "__main__":
    unittest.main()
