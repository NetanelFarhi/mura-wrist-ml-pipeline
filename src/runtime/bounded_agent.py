"""Bounded ambiguity investigation for the local Wrist Review prototype.

This module implements the data-contract shape proposed for the future Agentic
AI layer without granting it clinical authority.  Deterministic safety and the
frozen Model B route run first.  Only supported studies with conflicting
view-level evidence can receive one extra, approved uncertainty/TTA check.

There is deliberately no LLM or external API in this version.  The module is a
traceable, local bridge from the implemented deterministic runtime to the
planned adaptive agent.  It must not be presented as a clinically validated or
fully adaptive agent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4


CONTRACT_VERSION = "1.0"
AGENT_VERSION = "bounded-ambiguity-v0.1"
MAX_TOOL_CALLS = 6

TTARunner = Callable[[], Mapping[str, Any]]


@dataclass(frozen=True)
class ToolResult:
    """A typed record for one local tool execution."""

    tool: str
    status: str
    version: str
    timestamp_utc: str
    payload: dict[str, Any]
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tool_result(tool: str, status: str, payload: Mapping[str, Any], failure_reason: str | None = None) -> ToolResult:
    return ToolResult(
        tool=tool,
        status=status,
        version="1.0",
        timestamp_utc=_timestamp(),
        payload=dict(payload),
        failure_reason=failure_reason,
    )


def _base_evidence(deterministic_result: Mapping[str, Any]) -> list[ToolResult]:
    """Convert deterministic evidence into the first four contract records."""

    anatomy = deterministic_result.get("anatomy")
    anatomy = anatomy if isinstance(anatomy, Mapping) else {}
    qc_results = deterministic_result.get("qc_results")
    qc_results = qc_results if isinstance(qc_results, Sequence) and not isinstance(qc_results, (str, bytes)) else []
    evidence = deterministic_result.get("view_evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    probabilities = deterministic_result.get("per_view_probabilities")
    probabilities = probabilities if isinstance(probabilities, Sequence) and not isinstance(probabilities, (str, bytes)) else []

    records = [
        _tool_result("input_qc", "OK", {"views": list(qc_results)}),
        _tool_result("support_check", "OK", dict(anatomy)),
        _tool_result(
            "primary_model",
            "OK" if deterministic_result.get("classification_performed") is True else "FAILED",
            {
                "performed": bool(deterministic_result.get("classification_performed")),
                "study_probability": evidence.get("mean_probability"),
                "threshold": evidence.get("threshold"),
                "per_view_probabilities": list(probabilities),
            },
            None if deterministic_result.get("classification_performed") is True else str(deterministic_result.get("reason")),
        ),
        _tool_result("view_evidence", "OK" if evidence.get("status") == "SUCCESS" else "FAILED", dict(evidence)),
    ]
    return records


def _safe_report(
    *,
    deterministic_result: Mapping[str, Any],
    trace_id: str,
    agent_status: str,
    agent_outcome: str | None,
    allowed_tools: Sequence[ToolResult],
    triggers: Sequence[str],
    missing_evidence: Sequence[str] = (),
    reason: str,
    final_outcome: str | None = None,
    policy_violation: bool = False,
) -> dict[str, Any]:
    deterministic_outcome = str(deterministic_result.get("deterministic_outcome", "NEEDS_HUMAN_REVIEW"))
    final = final_outcome or deterministic_outcome
    return {
        "contract_version": CONTRACT_VERSION,
        "agent_version": AGENT_VERSION,
        "trace_id": trace_id,
        "agent_status": agent_status,
        "deterministic_outcome": deterministic_outcome,
        "agent_outcome": agent_outcome,
        "final_outcome": final,
        "triggers_fired": list(triggers),
        "allowed_tools_called": [record.to_dict() for record in allowed_tools],
        "missing_evidence": list(missing_evidence),
        "reason_for_escalation_or_rejection": reason,
        "policy_violation": bool(policy_violation),
        "human_review_required": True,
        "limitations": [
            "The current bounded agent is deterministic and local; no LLM is connected.",
            "It does not diagnose, change the model score, change the threshold, or clear a case from human review.",
        ],
    }


def run_bounded_agent(
    deterministic_result: Mapping[str, Any],
    *,
    tta_runner: TTARunner | None,
) -> dict[str, Any]:
    """Create a constrained case report after deterministic routing.

    The one allowed ambiguity trigger in this first implementation is
    disagreement across views around the frozen threshold.  The only optional
    call is a locally executed TTA stability check.  Its output is evidence for
    the reviewer; it cannot modify the deterministic route.
    """

    trace_id = uuid4().hex
    records = _base_evidence(deterministic_result)
    outcome = str(deterministic_result.get("deterministic_outcome", "NEEDS_HUMAN_REVIEW"))
    evidence = deterministic_result.get("view_evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}

    valid_supported = (
        deterministic_result.get("classification_performed") is True
        and outcome in {"RAISE_PRIORITY", "STANDARD_HUMAN_REVIEW"}
        and evidence.get("status") == "SUCCESS"
    )
    disagreement = bool(evidence.get("threshold_disagreement", False))
    if not valid_supported:
        return _safe_report(
            deterministic_result=deterministic_result,
            trace_id=trace_id,
            agent_status="not_called_deterministic_route",
            agent_outcome=None,
            allowed_tools=records,
            triggers=(),
            reason="deterministic_route_not_eligible_for_agent",
        )
    if not disagreement:
        return _safe_report(
            deterministic_result=deterministic_result,
            trace_id=trace_id,
            agent_status="not_called_no_ambiguity",
            agent_outcome=None,
            allowed_tools=records,
            triggers=(),
            reason="no_approved_ambiguity_trigger",
        )

    triggers = ["view_threshold_disagreement"]
    if tta_runner is None:
        return _safe_report(
            deterministic_result=deterministic_result,
            trace_id=trace_id,
            agent_status="safe_failure",
            agent_outcome="NEEDS_HUMAN_REVIEW",
            allowed_tools=records,
            triggers=triggers,
            missing_evidence=("uncertainty_tta_unavailable",),
            reason="agent_tool_unavailable",
            final_outcome="NEEDS_HUMAN_REVIEW",
        )

    try:
        tta = dict(tta_runner())
        required = {"status", "original_probability", "tta_mean_probability", "tta_std", "tta_range", "threshold_disagreement", "threshold"}
        if not required.issubset(tta) or tta.get("status") != "SUCCESS":
            raise ValueError("invalid_tta_result")
        records.append(_tool_result("uncertainty_tta", "OK", tta))
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        records.append(_tool_result("uncertainty_tta", "FAILED", {}, str(exc)))
        return _safe_report(
            deterministic_result=deterministic_result,
            trace_id=trace_id,
            agent_status="safe_failure",
            agent_outcome="NEEDS_HUMAN_REVIEW",
            allowed_tools=records,
            triggers=triggers,
            missing_evidence=("valid_uncertainty_tta_result",),
            reason="agent_tool_failure",
            final_outcome="NEEDS_HUMAN_REVIEW",
        )

    if len(records) > MAX_TOOL_CALLS:
        return _safe_report(
            deterministic_result=deterministic_result,
            trace_id=trace_id,
            agent_status="safe_failure",
            agent_outcome="NEEDS_HUMAN_REVIEW",
            allowed_tools=records,
            triggers=triggers,
            reason="agent_tool_budget_exceeded",
            final_outcome="NEEDS_HUMAN_REVIEW",
            policy_violation=True,
        )

    return _safe_report(
        deterministic_result=deterministic_result,
        trace_id=trace_id,
        agent_status="completed_bounded_investigation",
        agent_outcome="NO_CHANGE",
        allowed_tools=records,
        triggers=triggers,
        reason="view_disagreement_investigated_with_tta",
    )
