"""Deterministic optional-tool authorization for the local wrist app.

This module does not classify a study and does not change the deterministic
workflow route. It only decides whether pre-approved research evidence tools
may run after a successful Model B result.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


APPROVED_OPTIONAL_TOOLS = ("tta_uncertainty", "gradcam_explainer")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _finite_probability(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def validate_policy_config(config: Mapping[str, Any], *, expected_threshold: float) -> dict[str, Any]:
    """Validate the small, explicit policy file before it controls any tool."""

    policy = _mapping(config)
    threshold = _finite_probability(policy.get("frozen_classification_threshold"))
    margin = _finite_probability(policy.get("borderline_probability_margin"))
    authorization = _mapping(policy.get("tool_authorization"))
    tools = authorization.get("authorized_tools")

    if threshold is None or not math.isclose(threshold, float(expected_threshold), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("optional_tool_policy_threshold_mismatch")
    if margin is None or margin <= 0.0 or margin >= 0.5:
        raise ValueError("optional_tool_policy_margin_invalid")
    if not isinstance(tools, list) or tuple(tools) != APPROVED_OPTIONAL_TOOLS:
        raise ValueError("optional_tool_policy_tools_invalid")
    if policy.get("borderline_definition") != "absolute_distance_from_frozen_threshold_less_than_or_equal_to_margin":
        raise ValueError("optional_tool_policy_definition_invalid")
    safety = _mapping(policy.get("safety"))
    if any(safety.get(key) is not False for key in (
        "tools_change_model_prediction",
        "tools_change_threshold",
        "tools_change_worklist_route",
    )):
        raise ValueError("optional_tool_policy_safety_invalid")
    if safety.get("human_review_required") is not True or safety.get("research_use_only") is not True:
        raise ValueError("optional_tool_policy_human_review_invalid")

    return {
        "version": str(policy.get("version", "unknown")),
        "threshold": threshold,
        "borderline_margin": margin,
        "authorized_tools": list(APPROVED_OPTIONAL_TOOLS),
    }


def evaluate_optional_tool_policy(
    deterministic_result: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a fail-closed authorization record for one deterministic result."""

    normalized_policy = _mapping(policy)
    threshold = _finite_probability(normalized_policy.get("threshold"))
    margin = _finite_probability(normalized_policy.get("borderline_margin"))
    evidence = _mapping(deterministic_result.get("view_evidence"))
    mean_probability = _finite_probability(evidence.get("mean_probability"))
    evidence_threshold = _finite_probability(evidence.get("threshold"))
    outcome = str(deterministic_result.get("deterministic_outcome", "NEEDS_HUMAN_REVIEW"))

    base = {
        "policy_version": str(normalized_policy.get("version", "unknown")),
        "research_use_only": True,
        "human_review_required": True,
        "classification_modified": False,
        "threshold_modified": False,
        "route_modified": False,
        "authorized_tools": [],
        "triggers": [],
        "tool_execution_allowed": False,
    }
    if threshold is None or margin is None:
        return {**base, "status": "SAFETY_BLOCK", "reason": "policy_configuration_invalid"}
    if deterministic_result.get("classification_performed") is not True:
        return {**base, "status": "SAFETY_BLOCK", "reason": "classification_not_completed"}
    if outcome not in {"RAISE_PRIORITY", "STANDARD_HUMAN_REVIEW"}:
        return {**base, "status": "SAFETY_BLOCK", "reason": "deterministic_route_not_eligible"}
    if deterministic_result.get("human_review_required") is not True:
        return {**base, "status": "SAFETY_BLOCK", "reason": "human_review_contract_missing"}
    if evidence.get("status") != "SUCCESS" or mean_probability is None:
        return {**base, "status": "SAFETY_BLOCK", "reason": "view_evidence_invalid"}
    if evidence_threshold is None or not math.isclose(evidence_threshold, threshold, rel_tol=0.0, abs_tol=1e-12):
        return {**base, "status": "SAFETY_BLOCK", "reason": "frozen_threshold_mismatch"}

    triggers: list[str] = []
    if deterministic_result.get("model_predicted_abnormal") is True:
        triggers.append("abnormal_mean_prediction")
    if evidence.get("threshold_disagreement") is True:
        triggers.append("view_threshold_disagreement")
    if abs(mean_probability - threshold) <= margin:
        triggers.append("borderline_mean_probability")

    if triggers:
        return {
            **base,
            "status": "AUTHORIZED",
            "reason": "optional_tools_authorized_by_deterministic_policy",
            "authorized_tools": list(APPROVED_OPTIONAL_TOOLS),
            "triggers": triggers,
            "tool_execution_allowed": True,
            "mean_probability": mean_probability,
            "frozen_threshold": threshold,
            "borderline_margin": margin,
        }
    return {
        **base,
        "status": "NO_TOOLS",
        "reason": "consistent_normal_without_optional_tool_trigger",
        "mean_probability": mean_probability,
        "frozen_threshold": threshold,
        "borderline_margin": margin,
    }


__all__ = [
    "APPROVED_OPTIONAL_TOOLS",
    "evaluate_optional_tool_policy",
    "validate_policy_config",
]
