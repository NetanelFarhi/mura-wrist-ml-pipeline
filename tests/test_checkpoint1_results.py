"""Safety checks for the aggregate-only Checkpoint 1 results publisher."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.build_checkpoint1_results import (
    _assert_internal_validation_path,
    _load_validation_selection,
    highest_threshold_for_sensitivity,
)


def test_publisher_rejects_test_named_input() -> None:
    with pytest.raises(ValueError, match="official-valid/test"):
        _assert_internal_validation_path(Path("local/test_predictions.csv"))


def test_high_sensitivity_policy_uses_highest_eligible_threshold() -> None:
    labels = np.asarray([1, 1, 0, 0])
    scores = np.asarray([0.90, 0.70, 0.60, 0.10])
    policy = highest_threshold_for_sensitivity(labels, scores, target=0.90)
    assert policy["threshold"] == pytest.approx(0.70)
    assert policy["sensitivity"] == pytest.approx(1.0)
    assert policy["specificity"] == pytest.approx(1.0)


def test_publisher_rejects_selection_record_that_used_test_data(tmp_path: Path) -> None:
    (tmp_path / "selection.json").write_text(
        '{"selection_data": "validation only", "test_used_for_selection": true}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not rule out test use"):
        _load_validation_selection(tmp_path, model_name="test model")
