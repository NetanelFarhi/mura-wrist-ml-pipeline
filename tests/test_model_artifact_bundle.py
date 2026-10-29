"""Synthetic tests for Model B artifact identity checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_model_artifact_bundle import BundleVerificationError, verify_bundle


def write_bundle_spec(path: Path, checkpoint_hash: str) -> None:
    path.write_text(
        json.dumps(
            {
                "bundle_id": "synthetic-model-b",
                "model": {
                    "operating_policy": "highest validation threshold with sensitivity >= 0.95",
                    "threshold": 0.25,
                },
                "required_files": {
                    "best_resnet18_study_level.pt": {"sha256": checkpoint_hash},
                    "run_config.json": {},
                    "selection.json": {},
                    "training_history.json": {},
                    "validation_metrics.json": {},
                    "validation_predictions.csv": {},
                    "validation_predictions_mean.csv": {},
                    "validation_predictions_max.csv": {},
                    "frozen_primary_operating_policy.json": {},
                },
            }
        ),
        encoding="utf-8",
    )


def write_valid_bundle(directory: Path) -> Path:
    checkpoint = directory / "best_resnet18_study_level.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()
    write_bundle_spec(directory / "bundle.json", checkpoint_hash)

    for name in (
        "run_config.json",
        "training_history.json",
        "validation_metrics.json",
        "validation_predictions.csv",
        "validation_predictions_mean.csv",
        "validation_predictions_max.csv",
    ):
        (directory / name).write_text("{}", encoding="utf-8")
    (directory / "selection.json").write_text(
        json.dumps(
            {
                "selection_data": "validation only",
                "test_used_for_selection": False,
                "threshold_selection": {
                    "selection_rule": "highest_threshold_with_sensitivity_at_least_target",
                    "target_sensitivity": 0.95,
                    "threshold": 0.25,
                },
            }
        ),
        encoding="utf-8",
    )
    (directory / "frozen_primary_operating_policy.json").write_text(
        json.dumps(
            {
                "source_split": "internal_validation",
                "study_aggregation": "mean_probability",
                "test_used_for_selection": False,
                "threshold": 0.25,
            }
        ),
        encoding="utf-8",
    )
    return directory / "bundle.json"


def test_canonical_style_bundle_is_accepted(tmp_path: Path) -> None:
    spec = write_valid_bundle(tmp_path)
    summary = verify_bundle(tmp_path, spec)
    assert summary["target_sensitivity"] == pytest.approx(0.95)
    assert summary["study_aggregation"] == "mean_probability"
    assert summary["official_mura_valid_read_for_selection"] is False


def test_unexpected_checkpoint_is_rejected(tmp_path: Path) -> None:
    spec = write_valid_bundle(tmp_path)
    (tmp_path / "best_resnet18_study_level.pt").write_bytes(b"different checkpoint")
    with pytest.raises(BundleVerificationError, match="checkpoint hash"):
        verify_bundle(tmp_path, spec)


def test_test_selected_policy_is_rejected(tmp_path: Path) -> None:
    spec = write_valid_bundle(tmp_path)
    (tmp_path / "selection.json").write_text(
        json.dumps(
            {
                "selection_data": "validation only",
                "test_used_for_selection": True,
                "threshold_selection": {
                    "selection_rule": "highest_threshold_with_sensitivity_at_least_target",
                    "target_sensitivity": 0.95,
                    "threshold": 0.25,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(BundleVerificationError, match="test use"):
        verify_bundle(tmp_path, spec)
