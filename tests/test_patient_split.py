"""Synthetic tests for deterministic, patient-safe XR_WRIST splitting."""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
import zlib
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.index_mura import build_combined_wrist_manifest, write_manifest  # noqa: E402
from src.data.split_patients import create_patient_splits, read_split_manifest, write_split_manifest  # noqa: E402
from scripts.audit_data import main as audit_main  # noqa: E402


def _grayscale_png(width: int = 2, height: int = 3) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes([128]) * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def _write_study(root: Path, source: str, patient: str, study: str) -> None:
    study_dir = root / source / "XR_WRIST" / patient / study
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "image1.png").write_bytes(_grayscale_png())


class PatientSplitTests(unittest.TestCase):
    def test_official_valid_is_sealed_and_train_val_is_patient_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "MURA-v1.1"
            _write_study(root, "train", "patient00001", "study1_negative")
            _write_study(root, "train", "patient00002", "study1_positive")
            _write_study(root, "train", "patient00003", "study1_negative")
            _write_study(root, "train", "patient00004", "study1_positive")
            _write_study(root, "valid", "patient99999", "study1_positive")

            manifest = build_combined_wrist_manifest(root)
            first = create_patient_splits(manifest, validation_fraction=0.5, seed=91)
            second = create_patient_splits(manifest, validation_fraction=0.5, seed=91)
            self.assertEqual(first, second)

            by_patient: dict[str, set[str]] = defaultdict(set)
            for row in first:
                by_patient[row.patient_id].add(row.final_split)
            self.assertTrue(all(len(final_splits) == 1 for final_splits in by_patient.values()))
            self.assertEqual(by_patient["patient99999"], {"test"})
            self.assertEqual(sum("val" in values for values in by_patient.values()), 2)
            self.assertEqual(sum("train" in values for values in by_patient.values()), 2)

    def test_manifest_and_split_csv_round_trip_has_exact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "MURA-v1.1"
            _write_study(root, "train", "patient00001", "study1_negative")
            _write_study(root, "train", "patient00002", "study1_positive")
            manifest = build_combined_wrist_manifest(root, source_splits=("train",))
            split_rows = create_patient_splits(manifest, validation_fraction=0.5, seed=7)
            manifest_path = Path(temporary) / "wrist_manifest.csv"
            split_path = Path(temporary) / "split_manifest.csv"
            write_manifest(manifest, manifest_path)
            write_split_manifest(split_rows, split_path)

            self.assertEqual(
                manifest_path.read_text(encoding="utf-8").splitlines()[0],
                "patient_id,study_id,image_path,label,source_split,num_views_in_study",
            )
            self.assertEqual(
                split_path.read_text(encoding="utf-8").splitlines()[0],
                "patient_id,study_id,final_split",
            )
            self.assertEqual(read_split_manifest(split_path), split_rows)

    def test_audit_cli_data_root_alias_generates_manifest_and_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "MURA-v1.1"
            _write_study(root, "train", "patient00001", "study1_negative")
            _write_study(root, "train", "patient00002", "study1_positive")
            _write_study(root, "valid", "patient99999", "study1_negative")
            manifest_path = Path(temporary) / "wrist_manifest.csv"
            split_path = Path(temporary) / "split_manifest.csv"
            report_path = Path(temporary) / "audit.json"

            exit_code = audit_main(
                [
                    "--data-root",
                    str(root),
                    "--manifest-output",
                    str(manifest_path),
                    "--split-output",
                    str(split_path),
                    "--output",
                    str(report_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(split_path.is_file())
            self.assertIn('"official_valid_is_sealed_test": true', report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
