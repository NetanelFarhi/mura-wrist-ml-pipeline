"""Contracts for deterministic optional-evidence authorization.

These tests use only synthetic scores. They ensure the app cannot silently widen
the agreed tool triggers or execute a tool after an invalid safety route.
"""

from __future__ import annotations

from src.runtime.agent_tool_policy import evaluate_optional_tool_policy, validate_policy_config


THRESHOLD = 0.263567715883255


def policy() -> dict[str, object]:
    return validate_policy_config(
        {
            "version": "test",
            "frozen_classification_threshold": THRESHOLD,
            "borderline_probability_margin": 0.05,
            "borderline_definition": "absolute_distance_from_frozen_threshold_less_than_or_equal_to_margin",
            "tool_authorization": {"authorized_tools": ["tta_uncertainty", "gradcam_explainer"]},
            "safety": {
                "tools_change_model_prediction": False,
                "tools_change_threshold": False,
                "tools_change_worklist_route": False,
                "human_review_required": True,
                "research_use_only": True,
            },
        },
        expected_threshold=THRESHOLD,
    )


def result(*, mean: float, abnormal: bool = False, disagreement: bool = False) -> dict[str, object]:
    return {
        "classification_performed": True,
        "deterministic_outcome": "RAISE_PRIORITY" if abnormal or disagreement else "STANDARD_HUMAN_REVIEW",
        "human_review_required": True,
        "model_predicted_abnormal": abnormal,
        "view_evidence": {
            "status": "SUCCESS",
            "mean_probability": mean,
            "threshold": THRESHOLD,
            "threshold_disagreement": disagreement,
        },
    }


def test_consistent_normal_outside_borderline_band_has_no_optional_tools():
    authorization = evaluate_optional_tool_policy(result(mean=0.10), policy=policy())

    assert authorization["status"] == "NO_TOOLS"
    assert authorization["authorized_tools"] == []
    assert authorization["tool_execution_allowed"] is False


def test_borderline_mean_authorizes_the_fixed_pair_of_tools():
    authorization = evaluate_optional_tool_policy(result(mean=THRESHOLD + 0.04), policy=policy())

    assert authorization["status"] == "AUTHORIZED"
    assert authorization["triggers"] == ["borderline_mean_probability"]
    assert authorization["authorized_tools"] == ["tta_uncertainty", "gradcam_explainer"]


def test_abnormal_and_disagreement_are_reported_as_distinct_policy_triggers():
    authorization = evaluate_optional_tool_policy(
        result(mean=0.40, abnormal=True, disagreement=True),
        policy=policy(),
    )

    assert authorization["status"] == "AUTHORIZED"
    assert authorization["triggers"] == ["abnormal_mean_prediction", "view_threshold_disagreement"]


def test_ineligible_safety_route_fails_closed_even_when_score_is_borderline():
    unsafe = result(mean=THRESHOLD)
    unsafe["classification_performed"] = False

    authorization = evaluate_optional_tool_policy(unsafe, policy=policy())

    assert authorization["status"] == "SAFETY_BLOCK"
    assert authorization["tool_execution_allowed"] is False
