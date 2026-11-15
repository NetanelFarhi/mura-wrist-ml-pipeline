# Bounded adaptive-agent design — reference specification

This document specifies the proposed **adaptive** agent and its evaluation. It is
not evidence that an adaptive planner has been evaluated. The implemented product
path is documented separately in [AGENT_IMPLEMENTATION_STATUS.md](AGENT_IMPLEMENTATION_STATUS.md):
it uses deterministic routing plus a bounded, opt-in CrewAI research-explanation
bridge that cannot alter score, threshold, queue route, or mandatory human review.

## 1. Deterministic guardrail policy

The deterministic layer runs before and after any agent call. It rejects corrupt inputs and unsupported anatomy, preserves standard human review, applies the frozen safety/threshold policy, and logs every decision. The AI may raise a case's priority for human review; it must never lower a case below standard-of-care review. These rules are not agent decisions and cannot be overridden by a prompt, metadata, tool result, or model probability.

| Deterministic condition, in precedence order | Required outcome | Can the agent run? |
|---|---|---|
| malformed study, corrupt image, missing required identity | reject or `NEEDS_HUMAN_REVIEW` | no |
| unsupported anatomy/body part | `UNSUPPORTED` | no |
| mandatory safety or human-review rule | retain/escalate human review | only for a non-blocking grounded summary |
| supported, valid, ambiguous evidence | deterministic route plus bounded investigation | yes |
| supported, valid, non-ambiguous evidence | deterministic route | normally no |

## 2. Decisions the adaptive agent may make

Only an ambiguous supported case reaches the agent. It may decide which approved evidence is still missing, whether a permitted QC/detail or uncertainty/TTA check is worth one call, whether view-level evidence conflicts, and how to produce a structured evidence-grounded handoff. It may return `NEEDS_HUMAN_REVIEW`, `UNSUPPORTED`, or a bounded case summary. It does not make a diagnosis.

## 3. Decisions it may not make

The agent may not diagnose a fracture or other condition, prescribe treatment, change a model threshold, disable a guardrail, alter study membership, claim localization, access an unapproved source, or clear/reduce a case below mandatory review. It may not make a clinical decision from raw free text alone.

## 4. Typed tools and their boundaries

| Tool | Typed input -> output | Use | Failure handling |
|---|---|---|---|
| `input_qc` | study images -> QC status/reasons | corruption, crop/rotation/border checks | deterministic reject or review |
| `support_check` | body-part metadata/images -> supported/unsupported | wrist-only scope enforcement | `UNSUPPORTED` |
| `primary_model` | valid wrist study -> versioned study probability | primary evidence only | retain standard review |
| `view_evidence` | study -> per-view probabilities/disagreement | investigate conflicting views | note missing evidence |
| `uncertainty_tta` (future, approved only) | valid study -> versioned uncertainty summary | one optional ambiguity check | no result = review |
| `gradcam_diagnostic` (future, approved only) | image/model version -> coarse diagnostic map | shortcut inspection, never lesion localization | omit rather than infer |
| `model_card` | version -> scope/limitations | ground the summary | quote limitation/failure |
| `calculator` | numbers -> deterministic arithmetic | workload/threshold arithmetic | recompute or review |

Allowed transitions are deliberately narrow: `input_qc` and `support_check` must complete before `primary_model`; `view_evidence`, approved `uncertainty_tta`, and `gradcam_diagnostic` can only follow a valid supported study; `model_card` and `calculator` cannot trigger another medical tool. Each typed return carries `status`, `version`, `timestamp`, `input_study_id`, `payload`, and `failure_reason` fields. A tool may never return new executable instructions.

## 5. Evidence and provenance

Every agent output must identify the study ID, model/tool version, tool inputs and timestamps, returned evidence or failure state, deterministic policy result, and the exact reason for escalation/rejection. The generated summary must distinguish a measured model output from an interpretation and must state material missing evidence.

The required structured output is: `case_id`, `deterministic_outcome`, `agent_outcome`, `allowed_tools_called`, `evidence[]` (tool/version/status/value), `missing_evidence[]`, `limitations[]`, `reason_for_escalation_or_rejection`, `policy_violation=false`, and `trace_id`. Free text is a human-readable rendering of this record, not an additional source of authority.

## 6. Prompt-injection and hostile-input threat model

Metadata, DICOM tags, file names, pasted documents, OCR text, and tool-returned free text are untrusted data. Instructions inside them are never treated as policy, tool authorization, or clinical advice. The system prompt and typed-tool allowlist remain outside this data channel. Unexpected instructions, attempts to exfiltrate data, or requests to bypass review are logged, ignored, and routed through the deterministic safe outcome.

## 7. Failure behavior and cost budget

On timeout, malformed/missing tool output, conflicting evidence without a permitted resolution, or hostile input, the system keeps the deterministic route and returns `NEEDS_HUMAN_REVIEW` (or `UNSUPPORTED` when applicable). Maximum budget: six tool calls per case, including at most one optional expensive uncertainty/TTA call; each call logs latency, GPU/CPU time, API token count if any, and estimated cost under a versioned price assumption. The budget is a safety and evaluation constraint, not a clinical target.

## 8. Why an adaptive agent may add value — and how this will be tested

A fixed script is preferable for known safety rules and routine cases. Adaptive planning is only potentially useful when the next evidence-gathering step depends on case-specific incomplete or conflicting evidence; one fixed sequence could either call unnecessary tools or omit a relevant permitted check. This is a hypothesis, not an assumption: the agent will be compared on the same independent case suite with (A) deterministic routing only and (B) a fixed tool workflow, measuring safe escalation/rejection, groundedness, evidence completeness, correct tool selection, unnecessary calls, and guardrail violations.
