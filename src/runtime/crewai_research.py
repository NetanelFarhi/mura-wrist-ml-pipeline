"""Optional CrewAI research-explanation bridge for Wrist Review.

The deterministic route runs before this module. This bridge can turn
already-computed, de-identified evidence into a research explanation for every
valid supported wrist study. Optional evidence tools remain policy-controlled:
the local TTA and Grad-CAM-summary tools are available only when the explicit
deterministic policy authorizes them. A consistent normal study therefore receives a
no-tools CrewAI explanation. It never sees
raw pixels, file names, local paths, patient identifiers, or labels; it cannot
alter Model B probabilities, the frozen threshold, queue ordering, or the
mandatory human-review requirement.

CrewAI is deliberately opt-in.  A local operator must explicitly set
``MURA_CREWAI_ENABLED=1`` and configure an LLM model before any external call
is attempted.  Without that opt-in, the app keeps its fully local workflow.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from importlib.util import find_spec
import json
import math
import os
from pathlib import Path
import re
import threading
from typing import Any
from uuid import uuid4


AGENT_VERSION = "crewai-research-explanation-v1.2"
CONTRACT_VERSION = "1.0"
MAX_ITERATIONS = 3
MAX_LLM_CALLS = 3
_EXECUTION_LOCK = threading.RLock()

Executor = Callable[[dict[str, Any]], str]


class LLMCallBudget:
    """Keep one CrewAI run from reserving more than the approved call limit."""

    def __init__(self, max_calls: int = MAX_LLM_CALLS) -> None:
        self.max_calls = max_calls
        self.reserved_calls = 0

    def reserve_call(self) -> bool:
        if self.reserved_calls >= self.max_calls:
            return False
        self.reserved_calls += 1
        return True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _spec_path(project_root: Path) -> Path:
    return project_root / "configs" / "agentic_ai" / "mura_agent_specification.json"


def research_agent_status(project_root: Path) -> dict[str, Any]:
    """Report whether an operator has explicitly enabled the CrewAI bridge.

    This function never imports CrewAI or contacts a model provider.  It is
    safe to call from the app status endpoint.
    """

    enabled = os.environ.get("MURA_CREWAI_ENABLED", "").strip() == "1"
    model_name = os.environ.get("MURA_CREWAI_LLM_MODEL", "").strip()
    status: dict[str, Any] = {
        "enabled": enabled,
        "ready": False,
        "agent_version": AGENT_VERSION,
        "mode": "optional_research_explanation",
        "max_iterations": MAX_ITERATIONS,
        "max_llm_calls": MAX_LLM_CALLS,
        "raw_images_shared": False,
        "reason": None,
    }
    if not enabled:
        status["reason"] = "disabled_by_operator"
        return status
    spec_path = _spec_path(project_root)
    if not spec_path.is_file():
        status["reason"] = "agent_specification_missing"
        return status
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        status["reason"] = "agent_specification_invalid"
        return status

    model = _mapping(spec.get("model"))
    if (
        model.get("threshold") != 0.263567715883255
        or model.get("aggregation") != "mean_probability"
        or _mapping(spec.get("safety_constraints")).get("no_agent_override_of_model_predictions") is not True
    ):
        status["reason"] = "agent_specification_policy_mismatch"
        return status

    status["specification_version"] = str(spec.get("version", "unknown"))
    if not model_name:
        status["reason"] = "llm_model_not_configured"
        return status
    if find_spec("crewai") is None:
        status["reason"] = "crewai_dependency_not_installed"
        return status

    status.update({"ready": True, "reason": "ready", "llm_model": model_name})
    return status


def _successful_tta(agent_report: Mapping[str, Any]) -> dict[str, Any] | None:
    for tool_result in _sequence(agent_report.get("allowed_tools_called")):
        tool = _mapping(tool_result)
        if tool.get("tool") == "uncertainty_tta" and tool.get("status") == "OK":
            payload = _mapping(tool.get("payload"))
            if payload.get("status") == "SUCCESS":
                return payload
    return None


def _successful_gradcam(agent_report: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read only a completed local Grad-CAM summary, never a pixel map."""

    for tool_result in _sequence(agent_report.get("allowed_tools_called")):
        tool = _mapping(tool_result)
        if tool.get("tool") == "gradcam_explainer" and tool.get("status") == "OK":
            payload = _mapping(tool.get("payload"))
            if payload.get("status") == "SUCCESS":
                return payload
    return None


def _authorized_tools(
    deterministic_result: Mapping[str, Any],
    bounded_report: Mapping[str, Any],
) -> tuple[list[str], str | None]:
    """Honor the explicit deterministic authorization record exactly.

    The app never decides on its own whether optional tools should be exposed
    to CrewAI. The fixed policy allows both tools together for an abnormal
    mean prediction, cross-view threshold disagreement, or a configured
    borderline study mean; a consistent normal study has no optional tools.
    """

    policy = _mapping(deterministic_result.get("optional_tool_policy"))
    policy_status = str(policy.get("status", "SAFETY_BLOCK"))
    if policy_status == "NO_TOOLS":
        return [], None
    authorized = _sequence(policy.get("authorized_tools"))
    if policy_status != "AUTHORIZED" or authorized != ["tta_uncertainty", "gradcam_explainer"]:
        return [], "optional_tool_policy_not_authorized"
    if bounded_report.get("agent_status") != "completed_bounded_investigation":
        return [], "local_optional_evidence_not_completed"
    if _successful_tta(bounded_report) is None:
        return [], "local_tta_evidence_missing"
    if _successful_gradcam(bounded_report) is None:
        return [], "local_gradcam_evidence_missing"
    return ["tta_uncertainty", "gradcam_explainer"], None


def _is_eligible(deterministic_result: Mapping[str, Any]) -> tuple[bool, str]:
    """Require valid deterministic routing; an optional tool is not required."""

    if deterministic_result.get("classification_performed") is not True:
        return False, "deterministic_classification_not_completed"
    if deterministic_result.get("deterministic_outcome") not in {"RAISE_PRIORITY", "STANDARD_HUMAN_REVIEW"}:
        return False, "deterministic_route_not_eligible"
    evidence = _mapping(deterministic_result.get("view_evidence"))
    if evidence.get("status") != "SUCCESS":
        return False, "view_evidence_not_completed"
    if deterministic_result.get("human_review_required") is not True:
        return False, "human_review_contract_missing"
    return True, "eligible"


def _safe_context(
    deterministic_result: Mapping[str, Any],
    bounded_report: Mapping[str, Any],
    authorized_tools: Sequence[str],
) -> dict[str, Any]:
    """Create the only context allowed to leave the local deterministic layer."""

    evidence = _mapping(deterministic_result.get("view_evidence"))
    probabilities = [float(value) for value in _sequence(deterministic_result.get("per_view_probabilities"))]
    qc_results = [_mapping(item) for item in _sequence(deterministic_result.get("qc_results"))]
    tta = _successful_tta(bounded_report)
    tta_is_authorized = "tta_uncertainty" in authorized_tools
    gradcam = _successful_gradcam(bounded_report)
    gradcam_is_authorized = "gradcam_explainer" in authorized_tools
    if tta_is_authorized and tta is None:
        raise ValueError("local_tta_evidence_missing")
    if gradcam_is_authorized and gradcam is None:
        raise ValueError("local_gradcam_evidence_missing")

    statuses = [str(item.get("status", "UNKNOWN")) for item in qc_results]
    study_probability = float(evidence["mean_probability"])
    threshold = float(evidence["threshold"])
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in [study_probability, threshold, *probabilities]):
        raise ValueError("non_finite_or_out_of_range_evidence")

    return {
        "study_alias": f"LOCAL_WRIST_{uuid4().hex[:12].upper()}",
        "research_use_only": True,
        "human_review_required": True,
        "classification": {
            "deterministic_outcome": str(deterministic_result["deterministic_outcome"]),
            "study_probability": round(study_probability, 6),
            "threshold": threshold,
            "view_count": int(evidence.get("view_count", len(probabilities))),
            "view_threshold_disagreement": bool(evidence.get("threshold_disagreement", False)),
        },
        "technical_qc": {
            "view_count": len(qc_results),
            "statuses": statuses,
            "all_views_nonblocking": all(status != "FAIL" for status in statuses),
        },
        "authorized_tools": list(authorized_tools),
        "authorized_evidence": ({
            **({"tta_uncertainty": {
                "original_probability": float(tta["original_probability"]),
                "tta_mean_probability": float(tta["tta_mean_probability"]),
                "tta_std": float(tta["tta_std"]),
                "tta_range": float(tta["tta_range"]),
                "threshold_disagreement": bool(tta["threshold_disagreement"]),
                "augmentation_count": int(tta.get("augmentation_count", 0)),
            }} if tta_is_authorized and tta is not None else {}),
            **({"gradcam_explainer": {
                "method": str(gradcam["method"]),
                "views_processed": int(gradcam["views_processed"]),
                "view_summaries": [
                    {
                        "view_index": int(_mapping(item).get("view_index", -1)),
                        "model_probability": float(_mapping(item).get("model_probability", 0.0)),
                        "visualization_generated_in_memory": bool(_mapping(item).get("visualization_generated_in_memory", False)),
                    }
                    for item in _sequence(gradcam.get("view_summaries"))
                ],
                "maps_persisted": False,
                "maps_shared_with_crewai": False,
                "limitation": "Qualitative model-attribution evidence only; not lesion localization.",
            }} if gradcam_is_authorized and gradcam is not None else {}),
        }),
        "data_boundary": "No image bytes, file names, local paths, labels, or patient identifiers are supplied.",
    }


def _base_report(
    deterministic_result: Mapping[str, Any],
    *,
    status: str,
    execution_status: str,
    reason: str | None,
    trace_id: str,
    authorized_tools: Sequence[str],
    llm_calls_reserved: int = 0,
) -> dict[str, Any]:
    outcome = str(deterministic_result.get("deterministic_outcome", "NEEDS_HUMAN_REVIEW"))
    return {
        "contract_version": CONTRACT_VERSION,
        "agent_version": AGENT_VERSION,
        "trace_id": trace_id,
        "timestamp_utc": _utc_now(),
        "agent_status": status,
        "execution_status": execution_status,
        "stop_reason": reason,
        "deterministic_outcome": outcome,
        "final_outcome": outcome,
        "human_review_required": True,
        "research_use_only": True,
        "model_prediction_modified": False,
        "threshold_modified": False,
        "route_modified": False,
        "authorized_tools": list(authorized_tools),
        "llm_calls_reserved": llm_calls_reserved,
        "llm_call_budget": MAX_LLM_CALLS,
        "data_boundary": "Only de-identified structured evidence is eligible for the research explanation; raw images stay local.",
    }


def _task_description(context: Mapping[str, Any]) -> str:
    prompt_context = dict(context)
    authorized_tools = _sequence(context.get("authorized_tools"))
    if set(authorized_tools) == {"tta_uncertainty", "gradcam_explainer"}:
        prompt_context["authorized_evidence"] = {
            "tta_uncertainty": "Available only through the assigned read-only tool.",
            "gradcam_explainer": "Available only through the assigned read-only summary tool; no image or pixel map is available.",
        }
        tool_instruction = (
            "Use each assigned read-only retrieval tool exactly once before writing "
            "the summary."
        )
    else:
        prompt_context["authorized_evidence"] = {}
        tool_instruction = (
            "No optional evidence tool is authorized for this routine study. "
            "Do not attempt a tool call."
        )
    return """Prepare a concise, research-only explanation of already-computed evidence.

""" + tool_instruction + """
Do not perform new inference. Do not provide a diagnosis, treatment
advice, or lesion location.
Do not modify or recommend modifying the frozen probability, threshold, or queue
route. Mandatory human review remains required.

Return only JSON with this exact schema:
{
  "summary": "plain-language evidence summary, maximum 500 characters",
  "evidence_used": ["allowed evidence item"],
  "limitations": ["research limitation"],
  "research_use_only": true,
  "human_review_required": true
}

De-identified context follows:\n""" + json.dumps(prompt_context, ensure_ascii=False, sort_keys=True)


def _extract_json_object(raw: str) -> dict[str, Any]:
    """Accept a single JSON object, including one fenced JSON block."""

    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("agent_response_not_object")
    return parsed


def _validated_summary(raw: str) -> dict[str, Any]:
    """Validate generated text before it can be shown as research evidence."""

    payload = _extract_json_object(raw)
    summary = payload.get("summary")
    evidence_used = payload.get("evidence_used")
    limitations = payload.get("limitations")
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 500:
        raise ValueError("invalid_research_summary")
    if not isinstance(evidence_used, list) or not all(isinstance(item, str) and item.strip() for item in evidence_used):
        raise ValueError("invalid_evidence_used")
    if not isinstance(limitations, list) or not all(isinstance(item, str) and item.strip() for item in limitations):
        raise ValueError("invalid_limitations")
    if payload.get("research_use_only") is not True or payload.get("human_review_required") is not True:
        raise ValueError("required_safety_flags_missing")
    if re.search(r"\b(diagnos(?:is|e|ed)|fracture|treat(?:ment|ing)?|medication|discharge)\b", summary, flags=re.IGNORECASE):
        raise ValueError("unsupported_clinical_claim")
    return {
        "summary": summary.strip(),
        "evidence_used": [item.strip() for item in evidence_used][:6],
        "limitations": [item.strip() for item in limitations][:6],
    }


def _run_live_crewai(
    context: dict[str, Any],
    model_name: str,
    budget: LLMCallBudget,
) -> tuple[str, dict[str, int]]:
    """Build a fresh one-study CrewAI instance with only authorized tools."""

    from crewai import Agent, Crew, LLM, Process, Task
    from crewai.tools import tool

    tool_calls = {"tta_uncertainty": 0, "gradcam_explainer": 0}
    tools: list[Any] = []
    if "tta_uncertainty" in _sequence(context.get("authorized_tools")):
        @tool("tta_uncertainty")
        def retrieve_precomputed_tta() -> str:
            """Retrieve only the local, precomputed TTA evidence for this study."""

            tool_calls["tta_uncertainty"] += 1
            return json.dumps(context["authorized_evidence"]["tta_uncertainty"], ensure_ascii=False)

        tools.append(retrieve_precomputed_tta)

    if "gradcam_explainer" in _sequence(context.get("authorized_tools")):
        @tool("gradcam_explainer")
        def retrieve_precomputed_gradcam_summary() -> str:
            """Retrieve only the local Grad-CAM summary; never an image or pixel map."""

            tool_calls["gradcam_explainer"] += 1
            return json.dumps(context["authorized_evidence"]["gradcam_explainer"], ensure_ascii=False)

        tools.append(retrieve_precomputed_gradcam_summary)

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    llm_kwargs: dict[str, Any] = {"model": model_name}
    if api_key:
        llm_kwargs["api_key"] = api_key
    llm = LLM(**llm_kwargs)
    agent = Agent(
        role="MURA Wrist Research Explanation Assistant",
        goal="Explain only authorized, precomputed research evidence for a valid wrist study.",
        backstory="You are a bounded research assistant. You cannot diagnose, alter Model B, or clear a study from human review.",
        llm=llm,
        tools=tools,
        allow_delegation=False,
        verbose=False,
        max_iter=MAX_ITERATIONS,
    )
    task = Task(
        description=_task_description(context),
        expected_output="A valid JSON object matching the requested research-only schema.",
        agent=agent,
    )
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
        memory=False,
        cache=False,
    )
    result = crew.kickoff()
    return str(getattr(result, "raw", result)), tool_calls


def _live_execution(context: dict[str, Any], model_name: str) -> tuple[str, int, dict[str, int]]:
    """Execute with the Notebook 09-style temporary CrewAI call-budget hook."""

    from crewai.hooks import (
        get_before_llm_call_hooks,
        register_before_llm_call_hook,
        unregister_before_llm_call_hook,
    )

    budget = LLMCallBudget()

    def budget_hook(_context: Any) -> bool:
        return budget.reserve_call()

    with _EXECUTION_LOCK:
        if tuple(get_before_llm_call_hooks()):
            raise RuntimeError("crewai_hook_registry_not_clean")
        registered = False
        try:
            register_before_llm_call_hook(budget_hook)
            registered = True
            output, tool_calls = _run_live_crewai(context, model_name, budget)
            authorized_tools = set(_sequence(context.get("authorized_tools")))
            for tool_name, calls in tool_calls.items():
                if tool_name in authorized_tools and calls != 1:
                    raise ValueError(f"{tool_name}_evidence_tool_not_called_exactly_once")
                if tool_name not in authorized_tools and calls != 0:
                    raise ValueError(f"unauthorized_{tool_name}_evidence_tool_called")
            return output, budget.reserved_calls, tool_calls
        finally:
            if registered:
                unregister_before_llm_call_hook(budget_hook)


def run_crewai_research_explanation(
    deterministic_result: Mapping[str, Any],
    bounded_report: Mapping[str, Any],
    *,
    project_root: Path,
    executor: Executor | None = None,
) -> dict[str, Any]:
    """Optionally request a bounded CrewAI explanation after local evidence.

    ``executor`` exists only for offline unit tests; the browser application
    never supplies it.  Real execution requires the explicit environment gate
    and a fresh CrewAI instance for each eligible study.
    """

    trace_id = uuid4().hex
    eligible, reason = _is_eligible(deterministic_result)
    if not eligible:
        return _base_report(
            deterministic_result,
            status="not_called",
            execution_status="NOT_STARTED",
            reason=reason,
            trace_id=trace_id,
            authorized_tools=(),
        )

    authorized_tools, authorization_failure = _authorized_tools(
        deterministic_result,
        bounded_report,
    )
    if authorization_failure is not None:
        return _base_report(
            deterministic_result,
            status="safe_failure",
            execution_status="INCOMPLETE",
            reason=authorization_failure,
            trace_id=trace_id,
            authorized_tools=(),
        )

    status = research_agent_status(project_root)
    if not status["ready"] and executor is None:
        return _base_report(
            deterministic_result,
            status="not_configured",
            execution_status="NOT_STARTED",
            reason=str(status["reason"]),
            trace_id=trace_id,
            authorized_tools=authorized_tools,
        )
    if executor is not None and not status["enabled"]:
        return _base_report(
            deterministic_result,
            status="not_configured",
            execution_status="NOT_STARTED",
            reason="disabled_by_operator",
            trace_id=trace_id,
            authorized_tools=authorized_tools,
        )

    try:
        context = _safe_context(deterministic_result, bounded_report, authorized_tools)
    except (KeyError, TypeError, ValueError) as exc:
        return _base_report(
            deterministic_result,
            status="safe_failure",
            execution_status="INCOMPLETE",
            reason=f"context_validation_failed:{type(exc).__name__}",
            trace_id=trace_id,
            authorized_tools=authorized_tools,
        )

    try:
        if executor is not None:
            raw = executor(context)
            reserved_calls = 0
            retrieved_tools = ["offline_test_executor"]
        else:
            raw, reserved_calls, tool_calls = _live_execution(context, str(status["llm_model"]))
            retrieved_tools = [name for name, calls in tool_calls.items() if calls]
        summary = _validated_summary(str(raw))
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _base_report(
            deterministic_result,
            status="safe_failure",
            execution_status="INCOMPLETE",
            reason=f"research_explanation_unavailable:{type(exc).__name__}",
            trace_id=trace_id,
            authorized_tools=authorized_tools,
        )

    report = _base_report(
        deterministic_result,
        status="completed_research_explanation",
        execution_status="COMPLETED",
        reason=None,
        trace_id=trace_id,
        authorized_tools=authorized_tools,
        llm_calls_reserved=reserved_calls,
    )
    report.update(summary)
    report["study_alias"] = context["study_alias"]
    report["tool_evidence_retrieved"] = retrieved_tools
    return report
