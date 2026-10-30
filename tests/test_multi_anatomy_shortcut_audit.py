"""Focused unit tests for the all-anatomy peripheral-pixel audit helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_multi_anatomy_shortcut_audit import (
    PREDICTION_TOLERANCE,
    apply_tensor_perturbation,
    assert_canonical_match,
    bootstrap_variant,
)


class MultiAnatomyShortcutAuditTests(unittest.TestCase):
    def test_outer_border_changes_only_expected_edges(self) -> None:
        import torch

        tensor = torch.ones((3, 20, 20), dtype=torch.float32)
        changed = apply_tensor_perturbation(tensor, "outer_border_black_5pct")
        # Five percent of 20 pixels is one pixel.  The interior is untouched.
        self.assertTrue(torch.equal(changed[:, 1:-1, 1:-1], tensor[:, 1:-1, 1:-1]))
        self.assertFalse(torch.equal(changed[:, 0, :], tensor[:, 0, :]))
        self.assertFalse(torch.equal(changed[:, :, 0], tensor[:, :, 0]))

    def test_corner_mask_preserves_central_pixels(self) -> None:
        import torch

        tensor = torch.ones((3, 30, 30), dtype=torch.float32)
        changed = apply_tensor_perturbation(tensor, "four_corner_mask_black_10pct")
        self.assertTrue(torch.equal(changed[:, 10:20, 10:20], tensor[:, 10:20, 10:20]))
        self.assertFalse(torch.equal(changed[:, :3, :3], tensor[:, :3, :3]))

    def test_canonical_and_padding_labels_do_not_modify_post_transform_tensor(self) -> None:
        import torch

        tensor = torch.randn((3, 12, 12), dtype=torch.float32)
        self.assertTrue(torch.equal(apply_tensor_perturbation(tensor, "canonical"), tensor))
        self.assertTrue(torch.equal(apply_tensor_perturbation(tensor, "padding_midgray"), tensor))

    def test_saved_canonical_predictions_must_match_identity_and_probability(self) -> None:
        rows = pd.DataFrame(
            {
                "patient_id": ["p1", "p2"],
                "study_id": ["XR_HAND/p1/a", "XR_HAND/p2/b"],
                "label": [0, 1],
                "num_views": [2, 3],
                "abnormal_probability": [0.1, 0.9],
            }
        )
        assert_canonical_match(rows, rows.copy())
        mismatch = rows.copy()
        mismatch.loc[0, "abnormal_probability"] += PREDICTION_TOLERANCE * 2
        with self.assertRaisesRegex(RuntimeError, "differ"):
            assert_canonical_match(rows, mismatch)

    def test_bootstrap_summary_is_patient_clustered_and_finite(self) -> None:
        detailed = pd.DataFrame(
            {
                "patient_id": ["p1", "p1", "p2"],
                "abs_probability_change": [0.0, 0.2, 0.1],
                "decision_changed": [False, True, False],
            }
        )
        report = bootstrap_variant(detailed, replicates=40, seed=42)
        self.assertEqual(set(report["metric"]), {"mean_abs_probability_change", "decision_flip_percent"})
        self.assertTrue(np.isfinite(report[["point_estimate", "lower_95_ci", "upper_95_ci"]].to_numpy(dtype=float)).all())
        self.assertTrue((report["bootstrap_unit"] == "patient").all())


if __name__ == "__main__":
    unittest.main()
