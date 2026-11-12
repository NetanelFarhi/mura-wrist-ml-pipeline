"""Tests for the bounded ambiguity investigation contract.

These tests use synthetic evidence only.  They do not load a model, images,
or labels, and they make no claim about clinical performance.
"""

from __future__ import annotations

from src.runtime.bounded_agent import run_bounded_agent


def supported_result(*, disagreement: bool, outcome: str = "RAISE_PRIORITY") -> dict[str, object]:
    return {
        "classification_performed": True,
        "deterministic_outcome": outcome,
        "reason": "synthetic_test_evidence",
        "anatomy": {"status": "SUPPORTED_WRIST", "anatomy": "XR_WRIST", "issues": []},
        "qc_results": [{"image_id": "view_1", "status": "PASS", "issues": []}],
        "per_view_probabilities": [0.24, 0.29],
        "view_evidence": {
            "status": "SUCCESS",
            "mean_probability": 0.265,
            "threshold": 0.263567715883255,
            "threshold_disagreement": disagreement,
        },
    }


def successful_tta() -> dict[str, object]:
    return {
        "status": "SUCCESS",
        "original_probability": 0.24,
        "tta_mean_probability": 0.245,
        "tta_std": 0.004,
        "tta_range": 0.011,
        "threshold_disagreement": False,
        "threshold": 0.263567715883255,
    }


def test_agent_does_not_run_when_no_approved_ambiguity_exists():
    report = run_bounded_agent(
        supported_result(disagreement=False),
        tta_runner=lambda: (_ for _ in ()).throw(AssertionError("TTA must not run")),
    )

    assert report["agent_status"] == "not_called_no_ambiguity"
    assert report["final_outcome"] == "RAISE_PRIORITY"
    assert report["human_review_required"] is True
    assert len(report["allowed_tools_called"]) == 4


def test_agent_adds_tta_evidence_but_cannot_change_deterministic_route():
    report = run_bounded_agent(supported_result(disagreement=True), tta_runner=successful_tta)

    assert report["agent_status"] == "completed_bounded_investigation"
    assert report["agent_outcome"] == "NO_CHANGE"
    assert report["deterministic_outcome"] == "RAISE_PRIORITY"
    assert report["final_outcome"] == "RAISE_PRIORITY"
    assert report["human_review_required"] is True
    assert report["allowed_tools_called"][-1]["tool"] == "uncertainty_tta"


def test_agent_fails_closed_when_tta_evidence_is_invalid():
    report = run_bounded_agent(supported_result(disagreement=True), tta_runner=lambda: {"status": "BAD"})

    assert report["agent_status"] == "safe_failure"
    assert report["final_outcome"] == "NEEDS_HUMAN_REVIEW"
    assert report["human_review_required"] is True
    assert report["allowed_tools_called"][-1]["status"] == "FAILED"


def test_agent_never_runs_after_a_deterministic_nonclassification_route():
    result = supported_result(disagreement=True)
    result["classification_performed"] = False
    result["deterministic_outcome"] = "NEEDS_HUMAN_REVIEW"

    report = run_bounded_agent(result, tta_runner=successful_tta)

    assert report["agent_status"] == "not_called_deterministic_route"
    assert report["final_outcome"] == "NEEDS_HUMAN_REVIEW"
