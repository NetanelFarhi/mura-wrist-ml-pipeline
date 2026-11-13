"""Tests for the bounded ambiguity investigation contract.

These tests use synthetic evidence only.  They do not load a model, images,
or labels, and they make no claim about clinical performance.
"""

from __future__ import annotations

from src.runtime.bounded_agent import run_bounded_agent


def supported_result(*, disagreement: bool, outcome: str = "RAISE_PRIORITY") -> dict[str, object]:
    authorized = outcome == "RAISE_PRIORITY"
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
        "model_predicted_abnormal": False,
        "human_review_required": True,
        "optional_tool_policy": {
            "status": "AUTHORIZED" if authorized else "NO_TOOLS",
            "authorized_tools": ["tta_uncertainty", "gradcam_explainer"] if authorized else [],
            "triggers": ["view_threshold_disagreement"] if authorized and disagreement else [],
            "tool_execution_allowed": authorized,
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


def successful_gradcam() -> dict[str, object]:
    return {
        "status": "SUCCESS",
        "method": "synthetic_gradcam",
        "views_processed": 1,
        "view_summaries": [{"view_index": 0}],
        "maps_persisted": False,
        "maps_shared_with_crewai": False,
    }


def test_agent_does_not_run_optional_tool_for_consistent_normal_study():
    report = run_bounded_agent(
        supported_result(disagreement=False, outcome="STANDARD_HUMAN_REVIEW"),
        tta_runner=lambda: (_ for _ in ()).throw(AssertionError("TTA must not run")),
        gradcam_runner=lambda: (_ for _ in ()).throw(AssertionError("Grad-CAM must not run")),
    )

    assert report["agent_status"] == "not_called_no_optional_tools"
    assert report["final_outcome"] == "STANDARD_HUMAN_REVIEW"
    assert report["human_review_required"] is True
    assert len(report["allowed_tools_called"]) == 4


def test_agent_runs_tta_for_an_abnormal_priority_study_without_view_disagreement():
    result = supported_result(disagreement=False)
    result["model_predicted_abnormal"] = True
    result["optional_tool_policy"]["triggers"] = ["abnormal_mean_prediction"]  # type: ignore[index]

    report = run_bounded_agent(result, tta_runner=successful_tta, gradcam_runner=successful_gradcam)

    assert report["agent_status"] == "completed_bounded_investigation"
    assert report["triggers_fired"] == ["abnormal_mean_prediction"]
    assert report["allowed_tools_called"][-2]["tool"] == "uncertainty_tta"


def test_agent_adds_authorized_evidence_but_cannot_change_deterministic_route():
    report = run_bounded_agent(
        supported_result(disagreement=True),
        tta_runner=successful_tta,
        gradcam_runner=successful_gradcam,
    )

    assert report["agent_status"] == "completed_bounded_investigation"
    assert report["agent_outcome"] == "NO_CHANGE"
    assert report["deterministic_outcome"] == "RAISE_PRIORITY"
    assert report["final_outcome"] == "RAISE_PRIORITY"
    assert report["human_review_required"] is True
    assert report["allowed_tools_called"][-2]["tool"] == "uncertainty_tta"
    assert report["allowed_tools_called"][-1]["tool"] == "gradcam_explainer"


def test_agent_fails_closed_when_tta_evidence_is_invalid():
    report = run_bounded_agent(
        supported_result(disagreement=True),
        tta_runner=lambda: {"status": "BAD"},
        gradcam_runner=successful_gradcam,
    )

    assert report["agent_status"] == "safe_failure"
    assert report["final_outcome"] == "NEEDS_HUMAN_REVIEW"
    assert report["human_review_required"] is True
    assert report["allowed_tools_called"][-1]["status"] == "FAILED"


def test_agent_never_runs_after_a_deterministic_nonclassification_route():
    result = supported_result(disagreement=True)
    result["classification_performed"] = False
    result["deterministic_outcome"] = "NEEDS_HUMAN_REVIEW"

    report = run_bounded_agent(result, tta_runner=successful_tta, gradcam_runner=successful_gradcam)

    assert report["agent_status"] == "not_called_deterministic_route"
    assert report["final_outcome"] == "NEEDS_HUMAN_REVIEW"
