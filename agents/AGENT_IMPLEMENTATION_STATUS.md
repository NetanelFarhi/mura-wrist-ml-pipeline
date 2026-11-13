# Agent implementation status

## What is connected now

The local Wrist Review app has a bounded, deterministic optional-evidence investigation
in `src/runtime/bounded_agent.py`. The app always runs the deterministic safety
route first. Optional local evidence is allowed only for a supported wrist study
with successful QC when the explicit policy is triggered: an abnormal mean prediction,
view scores on both sides of the frozen threshold, or a mean inside the configurable
borderline band. A consistent normal study receives `NO_TOOLS`.

The current workflow records these typed evidence steps:

1. `input_qc`
2. `support_check`
3. `primary_model`
4. `view_evidence`
5. `uncertainty_tta` only when the fixed policy authorizes it
6. `gradcam_explainer` only when the same fixed policy authorizes it

The TTA check applies four small in-memory perturbations to the view closest to
the threshold: brightness -5%, brightness +5%, contrast -5%, and contrast +5%.
Its result is evidence for the reviewer only. It does not retrain a model,
change a probability, choose a new threshold, or remove a study from human
review.

The Grad-CAM tool runs against Model B's final ResNet block and returns a small,
in-memory per-view attribution summary. The heatmaps are not persisted or given to
CrewAI. They are qualitative model-attribution evidence, not validated lesion
localization.

The app also includes an **optional CrewAI research-explanation bridge** in
`src/runtime/crewai_research.py`. It is disabled by default and runs for every
valid supported study. A consistent normal study enters CrewAI with no optional
tool (`NO_TOOLS` behavior from Notebook 09). When the fixed policy authorizes
optional evidence, both app-connected read-only tools retrieve already-computed local
TTA and Grad-CAM-summary results. When an operator enables the bridge with local
provider configuration, it builds a fresh Agent, Task, and Crew for that one
study, with `max_iter=3`, delegation disabled, and a temporary three-call
reservation budget. It does not receive images or perform model inference.

The CrewAI input is a generated local study alias plus structured QC/model/evidence
values. It excludes raw pixels, file names, local paths, labels, patient
identifiers. Its response must satisfy a small JSON contract requiring
research-only and human-review flags. A configuration, call, budget, or schema
failure produces a safe incomplete report and leaves the deterministic outcome
unchanged.

## Explicit limits of the implemented workflow

- The local bounded layer is deterministic and uses no external API. The optional
  CrewAI layer is disabled unless an operator explicitly enables it.
- The CrewAI bridge is an evidence explanation path, not open-ended planning. A
  consistent normal case has no optional tool; an authorized case can retrieve
  only precomputed local TTA and Grad-CAM-summary evidence in the app integration.
- It uses no official MURA labels for selection or prompting.
- It cannot diagnose, localize an abnormality, prescribe action, or override
  the deterministic route.
- A missing or malformed authorized TTA or Grad-CAM result fails closed to
  `NEEDS_HUMAN_REVIEW`.
- Live provider retries and billing are not independently verified merely by the
  offline call-budget test. Do not present the bridge as clinically validated.

## What is still required before claiming an adaptive agent

1. Validate the configured live provider path and call-budget behavior with a
   pre-specified offline-safe test harness before claiming enforcement.
2. Implement any remaining approved tools only behind the deterministic guards.
3. Freeze a fixed-tool comparator and an independent, pre-specified case suite.
4. Evaluate deterministic routing, fixed tools, and the optional CrewAI routing on the same
   cases for safe routing, evidence completeness, unnecessary calls, costs, and
   guardrail violations.
5. Record the exact agent version in any final official-validation protocol.

The design constraints and the planned evaluation remain in
[AGENT_DESIGN.md](AGENT_DESIGN.md) and
[AGENT_EVAL_PLAN.md](../evaluation/AGENT_EVAL_PLAN.md).
