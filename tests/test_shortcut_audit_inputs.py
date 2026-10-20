"""Synthetic safety tests for the development-only shortcut audit."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_shortcut_audit.py"
SPEC = importlib.util.spec_from_file_location("run_shortcut_audit", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def _clean_rows(patient: str, study: str, label: str, split: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [patient, patient],
            "study_uid": [study, study],
            "file_name": [
                f"MURA-v1.1/train/XR_WRIST/{patient}/{study}/image1.png",
                f"MURA-v1.1/train/XR_WRIST/{patient}/{study}/image2.png",
            ],
            "label": [label, label],
            "split": [split, split],
        }
    )


class ShortcutAuditInputTests(unittest.TestCase):
    def test_separate_clean_manifests_are_train_only_and_patient_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path = root / "internal_train.csv"
            validation_path = root / "internal_validation.csv"
            _clean_rows("p-train", "s-train", "negative", "train").to_csv(train_path, index=False)
            _clean_rows("p-validation", "s-validation", "positive", "validation").to_csv(
                validation_path, index=False
            )
            args = SimpleNamespace(
                train_manifest=train_path,
                validation_manifest=validation_path,
                development_manifest=None,
                split_manifest=None,
            )
            train, validation, provenance = AUDIT.load_development_frames(args)
            self.assertEqual(len(train), 2)
            self.assertEqual(len(validation), 2)
            self.assertEqual(train["label"].tolist(), [0, 0])
            self.assertEqual(validation["label"].tolist(), [1, 1])
            self.assertEqual(provenance["input_contract"], "separate_clean_manifests")

    def test_official_valid_path_is_rejected_before_image_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unsafe.csv"
            unsafe = _clean_rows("p1", "s1", "negative", "train")
            unsafe.loc[0, "file_name"] = "MURA-v1.1/valid/XR_WRIST/p1/s1/image1.png"
            unsafe.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "train-only XR_WRIST"):
                AUDIT.standardize_development_manifest(path)

    def test_non_train_source_population_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unsafe_source.csv"
            unsafe = _clean_rows("p1", "s1", "negative", "train")
            unsafe["source_split"] = "valid"
            unsafe.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "non-train source_split"):
                AUDIT.standardize_development_manifest(path)

    def test_split_manifest_with_test_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "development.csv"
            split_path = root / "split.csv"
            development = pd.concat(
                [
                    _clean_rows("p1", "s1", "negative", "train"),
                    _clean_rows("p2", "s2", "positive", "train"),
                ],
                ignore_index=True,
            )
            development.to_csv(manifest_path, index=False)
            pd.DataFrame(
                {
                    "patient_id": ["p1", "p2"],
                    "study_id": ["s1", "s2"],
                    "final_split": ["train", "test"],
                }
            ).to_csv(split_path, index=False)
            args = SimpleNamespace(
                train_manifest=None,
                validation_manifest=None,
                development_manifest=manifest_path,
                split_manifest=split_path,
            )
            with self.assertRaisesRegex(ValueError, "never test"):
                AUDIT.load_development_frames(args)

    def test_patient_overlap_between_explicit_partitions_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path = root / "internal_train.csv"
            validation_path = root / "internal_validation.csv"
            _clean_rows("p1", "s1", "negative", "train").to_csv(train_path, index=False)
            _clean_rows("p1", "s2", "positive", "validation").to_csv(validation_path, index=False)
            args = SimpleNamespace(
                train_manifest=train_path,
                validation_manifest=validation_path,
                development_manifest=None,
                split_manifest=None,
            )
            with self.assertRaisesRegex(ValueError, "Patient overlap"):
                AUDIT.load_development_frames(args)

    def test_no_manifest_contract_has_no_repository_fallback(self) -> None:
        args = SimpleNamespace(
            train_manifest=None,
            validation_manifest=None,
            development_manifest=None,
            split_manifest=None,
        )
        with self.assertRaisesRegex(ValueError, "repository defaults"):
            AUDIT.load_development_frames(args)


if __name__ == "__main__":
    unittest.main()
