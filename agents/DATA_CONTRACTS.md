# Data contracts v1 — tool results, agent decision, case report

**Status:** proposal (29.10.2026), not implemented, not yet agreed by the team. Companion to `agents/DETERMINISTIC_POLICY_PROPOSAL.md` (rev. 2): that file defines *what* is decided; this file defines *how* the data is passed. Basis: `agents/AGENT_DESIGN.md` §1–7 (main @ 77f6ebc).

## 0. Principles

1. Three objects: `ToolResult` (tool → orchestrator), `AgentDecision` (LLM → code), `CaseReport` (code → clinician and audit log).
2. The LLM fills only `AgentDecision`. Every field of `CaseReport`, except the interpretation part of the summary, is filled by code.
3. The agent sees a pseudonymous `case_id`. Code adds `study_id` to `CaseReport` after the agent step (AGENT_DESIGN §5 requires the study ID in the output, not in the LLM context).
4. Time: ISO 8601, UTC. Contract version: `contract_version = "1.0"`.

## 1. ToolResult (AGENT_DESIGN §4)

| Field | Type | Meaning |
|---|---|---|
| `tool` | enum | `input_qc`, `support_check`, `primary_model`, `view_evidence`, `model_card`, `calculator` (`uncertainty_tta`, `gradcam_diagnostic` after CP2) |
| `status` | enum | `OK` / `FAILED` — **whether the tool itself ran**, not a verdict on the image |
| `version` | str | tool version; for the model, `model_version` (sha256 of weights) |
| `timestamp` | str | response time |
| `input_study_id` | str | = `case_id` (pseudonym) |
| `input_ref` | str | sha256 of the input and call parameters (§5: "tool inputs") |
| `payload` | object / null | typed result (§2); `null` when `FAILED` |
| `failure_reason` | str / null | reason code when `FAILED` |

- Any `FAILED`, or a response that fails schema validation → deterministic `NEEDS_HUMAN_REVIEW` (AGENT_DESIGN §4, §7).
- Exception: `view_evidence` with `n_views = 1` is not a failure; it returns `range = null`.
- Example of the status/verdict split: `input_qc` with `status = OK` and `verdict = FAIL` → `REJECTED`; `input_qc` with `status = FAILED` → `NEEDS_HUMAN_REVIEW`.
- Cost data (latency, CPU/GPU time, tokens, estimated cost, price version) goes to the JSONL trace, not to `ToolResult`.

## 2. Payload per tool

| Tool | Payload fields |
|---|---|
| `input_qc` | `verdict` (`PASS` / `SOFT_FLAG` / `FAIL`), `fail_reasons[]`, `soft_flags[]`, `n_images`, `duplicates_removed[]` (sha256) |
| `support_check` | `verdict` (`SUPPORTED` / `UNSUPPORTED` / `MIXED`), `body_parts[]` |
| `primary_model` | `p`, `threshold`, `view_scores[]` |
| `view_evidence` | `n_views`, `range`, `sd`, `lovo_delta[]` (change in p when one view is left out) |
| `model_card` | `scope`, `limitations[]`, `metrics_ref` |
| `calculator` | `operation` (fixed list: `margin_to_threshold`, `range`, `share`), `result` |

- `fail_reasons`: `DECODE_FAIL`, `BLANK_IMAGE`, `ID_MISSING`, `ID_MISMATCH`, `ZERO_IMAGES`, `TOO_MANY_IMAGES`, `INVALID_METADATA`.
- `soft_flags`: `OVEREXPOSED`, `UNDEREXPOSED`, `BLUR`, `CROP`, `ROTATION` — only those that are measurable.
- Codes, not free text: they can be counted and cannot carry injected instructions.

## 3. AgentDecision (LLM output)

| Field | Type | Constraint |
|---|---|---|
| `proposed_outcome` | enum | `NO_CHANGE` / `NEEDS_HUMAN_REVIEW` / `UNSUPPORTED` (AGENT_DESIGN §2) |
| `evidence_refs[]` | list | indices into `evidence[]`; `UNSUPPORTED` requires a reference to a `support_check` result with `verdict` ≠ `SUPPORTED` |
| `missing_evidence[]` | list | what was missing for a conclusion |
| `interpretation_text` | str | interpretation only; every statement references `evidence_refs` |

- Measured values (p, threshold, per-view scores, QC) are rendered by a code template; the LLM does not restate them (§5: measured output vs interpretation).
- Invalid JSON or timeout → `NEEDS_HUMAN_REVIEW`; no retry in v1 (§7).

## 4. CaseReport (AGENT_DESIGN §5)

| Field | Filled by | Type / values |
|---|---|---|
| `contract_version`, `policy_version`, `model_version` | code | str |
| `case_id`, `study_id` | code | pseudonym and source ID |
| `deterministic_outcome` | code | one of the 5 v1 outcomes |
| `model_result` | code | `{p, threshold, side: below_threshold / above_threshold}` or `null` if the model did not run |
| `triggers_fired[]` | code | `[1–3]`; empty → agent not called |
| `agent_outcome` | code (from `AgentDecision`) | `NO_CHANGE` / `NEEDS_HUMAN_REVIEW` / `UNSUPPORTED` or `null` |
| `final_outcome` | code | `deterministic_outcome` if the agent was not called or returned `NO_CHANGE`; otherwise `agent_outcome`. Agent `UNSUPPORTED` without tool backing → `NEEDS_HUMAN_REVIEW` |
| `allowed_tools_called[]` | code | `{tool, version, status, timestamp}` |
| `evidence[]` | code | `{tool, version, status, value}` from `ToolResult` |
| `missing_evidence[]` | code + agent | union |
| `limitations[]` | code | from `model_card` + fixed flag wording |
| `reason_for_escalation_or_rejection` | code | reason codes: §2 codes + `NOT_WRIST`, `MIXED_BODY_PARTS`, `TOOL_FAILURE`, `AGENT_ESCALATION`, `AGENT_TIMEOUT`, `AGENT_INVALID_OUTPUT`, `UNGROUNDED_UNSUPPORTED` |
| `policy_violation` | code | `true` on tool-unbacked `UNSUPPORTED`, a reference to non-existent evidence, or budget overrun; final outcome is then `NEEDS_HUMAN_REVIEW` |
| `summary_text` | code + agent | measured block (template) + `interpretation_text`; template only on violation or failure |
| `trace_id` | code | link to the JSONL trace |

## 5. Case flow through the contracts

1. Code calls `input_qc` and `support_check` → two `ToolResult`. Defect or non-wrist → `CaseReport` with `REJECTED` / `UNSUPPORTED`; stop.
2. Code calls `primary_model` and `view_evidence` → `p` and per-view scores; threshold gives `NOT_FLAGGED` or `FLAGGED_REVIEW`.
3. Code checks triggers 1–3. None fired → `CaseReport` with template summary.
4. A trigger fired → the LLM receives the evidence (no IDs, no pixels) and returns `AgentDecision`.
5. Code validates `AgentDecision` (schema and references), computes `final_outcome`, writes `CaseReport`.

## 6. Conformance to AGENT_DESIGN

| Section | Status |
|---|---|
| §1 precedence and rows | conforms (policy proposal §2) |
| §2 agent outcomes | conforms: `NEEDS_HUMAN_REVIEW`, `UNSUPPORTED`, bounded summary (`NO_CHANGE`) |
| §4 tools, fields, transitions | conforms; `calculator` kept with a fixed operation set |
| §5 fields, study ID, measured vs interpretation | conforms |
| §6 hostile input | metadata outside the vocabulary → `NEEDS_HUMAN_REVIEW`; no free text is passed to the LLM |
| §7 failures and budget | conforms; reading "6 calls per case incl. pipeline" to be agreed |

## 7. Risks

- Contracts protect only once implemented (e.g. Pydantic or JSON Schema) and covered by tests.
- A contract checks form, not truth: a wrong `p` passes validation.
- Without reason codes in the clinician UI, `NEEDS_HUMAN_REVIEW` with different causes looks the same.
- Changing the contract after coding starts requires changes in every component.
