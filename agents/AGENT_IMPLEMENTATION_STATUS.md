# Agent implementation status

## What is connected now

The local Wrist Review app has a bounded, deterministic ambiguity investigation
in `src/runtime/bounded_agent.py`. The app always runs the deterministic safety
route first. Only a supported wrist study with successful QC and view scores on
both sides of the frozen threshold can enter the additional investigation.

The current workflow records these typed evidence steps:

1. `input_qc`
2. `support_check`
3. `primary_model`
4. `view_evidence`
5. `uncertainty_tta` only when the approved ambiguity trigger fires

The TTA check applies four small in-memory perturbations to the view closest to
the threshold: brightness -5%, brightness +5%, contrast -5%, and contrast +5%.
Its result is evidence for the reviewer only. It does not retrain a model,
change a probability, choose a new threshold, or remove a study from human
review.

## Explicit limits of version 0.1

- It is not an LLM or LangGraph agent.
- It uses no external API, prompt, patient data source, or official MURA labels.
- It has one optional tool call, not open-ended tool selection.
- It cannot diagnose, localize an abnormality, prescribe action, or override
  the deterministic route.
- A missing or malformed TTA result fails closed to `NEEDS_HUMAN_REVIEW`.

## What is still required before claiming an adaptive agent

1. Choose and version an approved planning runtime and model/provider.
2. Implement the remaining approved tools only behind the deterministic guards.
3. Freeze a fixed-tool comparator and an independent, pre-specified case suite.
4. Evaluate deterministic routing, fixed tools, and adaptive routing on the same
   cases for safe routing, evidence completeness, unnecessary calls, costs, and
   guardrail violations.
5. Record the exact agent version in any final official-validation protocol.

The design constraints and the planned evaluation remain in
[AGENT_DESIGN.md](AGENT_DESIGN.md) and
[AGENT_EVAL_PLAN.md](../evaluation/AGENT_EVAL_PLAN.md).
