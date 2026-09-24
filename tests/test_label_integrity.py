"""Synthetic integrity tests for labels, missing/corrupt images, and dimensions."""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_data import audit_manifest  # noqa: E402
from src.data.split_patients import validate_manifest_studies  # noqa: E402


def _grayscale_png(width: int = 3, height: int = 2) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes([96]) * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def _row(patient: str, study: str, image: str, label: int, views: int = 1) -> dict[str, object]:
    return {
        "patient_id": patient,
        "study_id": f"{patient}/{study}",
        "image_path": f"train/XR_WRIST/{patient}/{study}/{image}",
        "label": label,
        "source_split": "train",
        "num_views_in_study": views,
    }


class LabelIntegrityTests(unittest.TestCase):
    def test_study_label_inconsistency_is_rejected_before_split(self) -> None:
        rows = [
            _row("patient1", "study1_positive", "view1.png", 1, views=2),
            _row("patient1", "study1_positive", "view2.png", 0, views=2),
        ]
        with self.assertRaisesRegex(ValueError, "inconsistent labels"):
            validate_manifest_studies(rows)

    def test_audit_reports_measured_data_quality_findings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "train/XR_WRIST/patient1/study1_negative/view1.png"
            valid.parent.mkdir(parents=True, exist_ok=True)
            valid.write_bytes(_grayscale_png())
            corrupt = root / "train/XR_WRIST/patient2/study1_positive/bad.png"
            corrupt.parent.mkdir(parents=True, exist_ok=True)
            corrupt.write_bytes(b"not an image")
            first_inconsistent = root / "train/XR_WRIST/patient4/study1_positive/view1.png"
            second_inconsistent = root / "train/XR_WRIST/patient4/study1_positive/view2.png"
            first_inconsistent.parent.mkdir(parents=True, exist_ok=True)
            first_inconsistent.write_bytes(_grayscale_png())
            second_inconsistent.write_bytes(_grayscale_png())

            rows = [
                _row("patient1", "study1_negative", "view1.png", 0),
                _row("patient1", "study1_negative", "view1.png", 0),  # intentional duplicate path
                _row("patient2", "study1_positive", "bad.png", 1),
                _row("patient3", "study1_negative", "missing.png", 0),
                _row("patient4", "study1_positive", "view1.png", 1, views=2),
                _row("patient4", "study1_positive", "view2.png", 0, views=2),
            ]
            report = audit_manifest(rows, dataset_root=root)

            self.assertEqual(report["counts"]["patients"], 4)
            self.assertEqual(report["counts"]["studies"], 4)
            self.assertEqual(len(report["duplicate_paths"]), 1)
            self.assertEqual(len(report["missing_files"]), 1)
            self.assertEqual(len(report["corrupt_images"]), 1)
            self.assertEqual(len(report["inconsistent_labels_inside_study"]), 1)
            self.assertIn("3x2", report["image_dimensions"]["by_width_x_height"])


if __name__ == "__main__":
    unittest.main()
