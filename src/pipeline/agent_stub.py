"""Placeholder for the agent layer (Checkpoint 1, until 18.10.2026).

The stub fixes the contract between the frozen image model and the future agent
layer: it takes one ``ModelOutput`` per study and returns one ``AgentReport``.
It has no LLM, no guardrails and no tools. It only restates the model flag
against the frozen threshold, so the pipeline can run end to end today and the
real agent (Checkpoint 2) can replace it without changing the interface.
"""
from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional, Tuple

STUB_LIMITATION = "agent layer not implemented: stub"
CONTRACT_VERSION = "1.0"

ModelFlag = Literal["ABOVE_THRESHOLD", "BELOW_THRESHOLD"]
AgentStatus = Literal["STUB"]


def _check_probability(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1], got {value!r}")
    return value


@dataclass(frozen=True)
class ModelOutput:
    """Stub input: the output of the frozen model for one study."""

    study_id: str
    model_version: str  # e.g. "model_a@<sha256[:12]>" or "model_b@<sha256[:12]>"
    study_probability: float
    threshold: float  # frozen operating threshold from operating_policy.json
    view_probabilities: Optional[Tuple[float, ...]] = None  # None for Model A

    def __post_init__(self) -> None:
        if not str(self.study_id).strip():
            raise ValueError("study_id must be a non-empty string")
        if not str(self.model_version).strip():
            raise ValueError("model_version must be a non-empty string")
        object.__setattr__(self, "study_probability", _check_probability("study_probability", self.study_probability))
        object.__setattr__(self, "threshold", _check_probability("threshold", self.threshold))
        if self.view_probabilities is not None:
            views = tuple(_check_probability("view_probability", v) for v in self.view_probabilities)
            if not views:
                raise ValueError("view_probabilities must be None or non-empty")
            object.__setattr__(self, "view_probabilities", views)


@dataclass(frozen=True)
class AgentReport:
    """Stub output: one report per study."""

    case_id: str
    model_flag: ModelFlag
    agent_status: AgentStatus  # at Checkpoint 2 this field carries the agent's decision
    evidence: ModelOutput  # the input, unchanged
    limitations: Tuple[str, ...] = field(default=(STUB_LIMITATION,))
    trace_id: str = ""
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> dict:
        record = asdict(self)
        evidence = record["evidence"]
        if evidence["view_probabilities"] is not None:
            evidence["view_probabilities"] = list(evidence["view_probabilities"])
        record["limitations"] = list(record["limitations"])
        return record

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


def agent_stub(x: ModelOutput, *, trace_id: Optional[str] = None) -> AgentReport:
    """Return the stub report for one study. No LLM, no guardrails, no tools.

    The flag uses the same rule as the evaluation notebooks:
    probability >= threshold -> ABOVE_THRESHOLD (flag for specialist review).
    """
    if not isinstance(x, ModelOutput):
        raise TypeError("agent_stub expects a ModelOutput")
    flag: ModelFlag = "ABOVE_THRESHOLD" if x.study_probability >= x.threshold else "BELOW_THRESHOLD"
    return AgentReport(
        case_id=x.study_id,
        model_flag=flag,
        agent_status="STUB",
        evidence=x,
        limitations=(STUB_LIMITATION,),
        trace_id=trace_id if trace_id is not None else uuid.uuid4().hex,
    )
