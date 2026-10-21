"""Regression tests for the lecturer-required Model A selection protocol."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.models.frozen_embedding_baseline import (
    FrozenEmbeddingLogisticBaseline,
    make_patient_grouped_folds,
)
from src.training.metrics import select_highest_threshold_for_sensitivity


def test_inner_cv_folds_keep_patients_disjoint() -> None:
    labels = np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64)
    patient_ids = np.asarray(["p1", "p1", "p2", "p3", "p4", "p5", "p6", "p7"])

    folds = make_patient_grouped_folds(labels, patient_ids, n_splits=2, random_seed=42)

    held_out = []
    for train_index, holdout_index in folds:
        assert not set(patient_ids[train_index]).intersection(patient_ids[holdout_index])
        held_out.extend(holdout_index.tolist())
    assert sorted(held_out) == list(range(len(labels)))


def test_baseline_fit_accepts_no_validation_data_for_c_selection() -> None:
    signature = inspect.signature(FrozenEmbeddingLogisticBaseline.fit)
    assert "validation_embeddings" not in signature.parameters
    assert "validation_labels" not in signature.parameters

    embeddings = np.asarray(
        [[0.0], [0.1], [0.2], [0.8], [0.9], [1.0], [0.05], [0.95]], dtype=np.float32
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1, 0, 1], dtype=np.int64)
    patient_ids = np.asarray(["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"])

    baseline = FrozenEmbeddingLogisticBaseline(c_values=(0.1, 1.0)).fit(
        embeddings, labels, patient_ids, group_cv_folds=2
    )
    metadata = baseline.selection_metadata()

    assert metadata["internal_validation_used_for_c_selection"] is False
    assert metadata["group_field"] == "patient_id"
    assert metadata["group_cv_folds"] == 2
    assert len(metadata["candidate_results"]) == 2


def test_high_sensitivity_policy_uses_highest_eligible_threshold_at_95_percent() -> None:
    labels = np.asarray([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    scores = np.asarray([0.99, 0.90, 0.80, 0.70, 0.60, 0.65, 0.55, 0.40, 0.30, 0.20])

    policy = select_highest_threshold_for_sensitivity(
        labels, scores, target_sensitivity=0.95
    )

    assert policy.threshold == pytest.approx(0.60)
    assert policy.achieved_sensitivity == pytest.approx(1.0)
    assert policy.achieved_specificity == pytest.approx(0.80)
