"""Offline contracts for the optional CrewAI research-explanation bridge.

No model, image, provider, or CrewAI installation is needed for these tests.
They exercise the same structured evidence boundary used by the local app.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.runtime.crewai_research import research_agent_status, run_crewai_research_explanation


def ambiguous_result() -> dict[str, object]:
    return {
        "case_id": "PRIVATE_CASE_NAME_MUST_NOT_LEAVE_LOCAL_RUNTIME",
        "classification_performed": True,
        "deterministic_outcome": "RAISE_PRIORITY",
        "human_review_required": True,
        "per_view_probabilities": [0.24, 0.29],
        "qc_results": [
            {"image_id": "private-file-a.png", "status": "PASS"},
            {"image_id": "private-file-b.png", "status": "PASS"},
        ],
        "view_evidence": {
            "status": "SUCCESS",
            "mean_probability": 0.265,
            "threshold": 0.263567715883255,
            "view_count": 2,
            "threshold_disagreement": True,
        },
        "optional_tool_policy": {
            "status": "AUTHORIZED",
            "authorized_tools": ["tta_uncertainty", "gradcam_explainer"],
            "triggers": ["view_threshold_disagreement", "borderline_mean_probability"],
            "tool_execution_allowed": True,
        },
    }


def completed_local_report() -> dict[str, object]:
    return {
        "agent_status": "completed_bounded_investigation",
        "allowed_tools_called": [
            {
                "tool": "uncertainty_tta",
                "status": "OK",
                "payload": {
                    "status": "SUCCESS",
                    "original_probability": 0.24,
                    "tta_mean_probability": 0.245,
                    "tta_std": 0.004,
                    "tta_range": 0.011,
                    "threshold_disagreement": False,
                    "augmentation_count": 4,
                },
            },
            {
                "tool": "gradcam_explainer",
                "status": "OK",
                "payload": {
                    "status": "SUCCESS",
                    "method": "gradcam_layer4_last_block_conv2",
                    "views_processed": 2,
                    "view_summaries": [
                        {"view_index": 0, "model_probability": 0.24, "visualization_generated_in_memory": True},
                        {"view_index": 1, "model_probability": 0.29, "visualization_generated_in_memory": True},
                    ],
                    "maps_persisted": False,
                    "maps_shared_with_crewai": False,
                },
            },
        ],
    }


def valid_agent_json() -> str:
    return json.dumps(
        {
            "summary": "The deterministic route and local stability evidence are retained for priority human review.",
            "evidence_used": ["frozen study probability", "precomputed local TTA stability evidence"],
            "limitations": ["Research-only explanation", "Human review remains required"],
            "research_use_only": True,
            "human_review_required": True,
        }
    )


def test_disabled_operator_setting_never_calls_an_executor(monkeypatch, tmp_path):
    monkeypatch.delenv("MURA_CREWAI_ENABLED", raising=False)
    called = False

    def executor(_context):
        nonlocal called
        called = True
        return valid_agent_json()

    report = run_crewai_research_explanation(
        ambiguous_result(),
        completed_local_report(),
        project_root=tmp_path,
        executor=executor,
    )

    assert report["agent_status"] == "not_configured"
    assert report["stop_reason"] == "disabled_by_operator"
    assert report["final_outcome"] == "RAISE_PRIORITY"
    assert called is False


def test_offline_executor_receives_only_deidentified_structured_evidence(monkeypatch, tmp_path):
    monkeypatch.setenv("MURA_CREWAI_ENABLED", "1")
    monkeypatch.setenv("MURA_CREWAI_LLM_MODEL", "offline-test")
    observed: dict[str, object] = {}

    def executor(context):
        observed.update(context)
        return valid_agent_json()

    report = run_crewai_research_explanation(
        ambiguous_result(),
        completed_local_report(),
        project_root=tmp_path,
        executor=executor,
    )

    serialized = json.dumps(observed)
    assert report["agent_status"] == "completed_research_explanation"
    assert report["execution_status"] == "COMPLETED"
    assert report["final_outcome"] == "RAISE_PRIORITY"
    assert report["model_prediction_modified"] is False
    assert report["threshold_modified"] is False
    assert report["route_modified"] is False
    assert report["human_review_required"] is True
    assert "PRIVATE_CASE_NAME" not in serialized
    assert "private-file-a.png" not in serialized
    assert observed["authorized_evidence"] == {
        "tta_uncertainty": {
            "original_probability": 0.24,
            "tta_mean_probability": 0.245,
            "tta_std": 0.004,
            "tta_range": 0.011,
            "threshold_disagreement": False,
            "augmentation_count": 4,
        },
        "gradcam_explainer": {
            "method": "gradcam_layer4_last_block_conv2",
            "views_processed": 2,
            "view_summaries": [
                {"view_index": 0, "model_probability": 0.24, "visualization_generated_in_memory": True},
                {"view_index": 1, "model_probability": 0.29, "visualization_generated_in_memory": True},
            ],
            "maps_persisted": False,
            "maps_shared_with_crewai": False,
            "limitation": "Qualitative model-attribution evidence only; not lesion localization.",
        },
    }


def test_invalid_agent_output_fails_closed_without_changing_route(monkeypatch, tmp_path):
    monkeypatch.setenv("MURA_CREWAI_ENABLED", "1")
    monkeypatch.setenv("MURA_CREWAI_LLM_MODEL", "offline-test")

    report = run_crewai_research_explanation(
        ambiguous_result(),
        completed_local_report(),
        project_root=tmp_path,
        executor=lambda _context: json.dumps(
            {
                "summary": "The study has a fracture.",
                "evidence_used": ["model"],
                "limitations": [],
                "research_use_only": True,
                "human_review_required": True,
            }
        ),
    )

    assert report["agent_status"] == "safe_failure"
    assert report["execution_status"] == "INCOMPLETE"
    assert report["final_outcome"] == "RAISE_PRIORITY"
    assert report["route_modified"] is False


def test_consistent_normal_study_runs_a_no_tools_research_explanation(monkeypatch, tmp_path):
    monkeypatch.setenv("MURA_CREWAI_ENABLED", "1")
    result = {
        "case_id": "PRIVATE_NORMAL_CASE",
        "classification_performed": True,
        "deterministic_outcome": "STANDARD_HUMAN_REVIEW",
        "human_review_required": True,
        "per_view_probabilities": [0.01, 0.02],
        "qc_results": [{"image_id": "private-normal.png", "status": "PASS"}],
        "view_evidence": {
            "status": "SUCCESS",
            "mean_probability": 0.015,
            "threshold": 0.263567715883255,
            "view_count": 2,
            "threshold_disagreement": False,
        },
        "optional_tool_policy": {
            "status": "NO_TOOLS",
            "authorized_tools": [],
            "triggers": [],
            "tool_execution_allowed": False,
        },
    }
    observed: dict[str, object] = {}

    def executor(context):
        observed.update(context)
        return valid_agent_json()

    report = run_crewai_research_explanation(
        result,
        {"agent_status": "not_called_no_optional_tools", "allowed_tools_called": []},
        project_root=tmp_path,
        executor=executor,
    )

    assert report["agent_status"] == "completed_research_explanation"
    assert report["authorized_tools"] == []
    assert observed["authorized_tools"] == []
    assert observed["authorized_evidence"] == {}


def test_status_is_read_only_when_operator_has_not_enabled_crewai(monkeypatch, tmp_path):
    monkeypatch.delenv("MURA_CREWAI_ENABLED", raising=False)
    status = research_agent_status(Path(tmp_path))

    assert status["enabled"] is False
    assert status["ready"] is False
    assert status["reason"] == "disabled_by_operator"
