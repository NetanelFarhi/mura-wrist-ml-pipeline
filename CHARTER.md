# DS23 Project Charter — MURA XR_WRIST Decision Support

**Team:** Netanel, Hagit, Hodaya, Artium
**Scope:** research prototype only; not a diagnostic or clinical-deployment system.

## 1. Problem + user

The primary user is an Emergency Department physician reviewing wrist radiographs before specialist radiology review is available. The project investigates a study-level ML system that flags **MURA-defined musculoskeletal abnormality** and supports human review.

MURA's label is **NORMAL / ABNORMAL**, not an acute-fracture diagnosis. The system will not claim fracture detection, lesion localization, autonomous diagnosis, treatment advice, or a measurable reduction in return visits. Any clinical motivation is background only, not a project outcome.

## 2. Success metric

**Primary target (pre-specified):** study-level sensitivity **>= 95%** under the defined development/evaluation protocol. For each model, choose the **highest** study-level probability threshold that achieves the target on the internal validation set, then freeze that model-specific threshold.

At that operating point, report specificity, false-negative rate (`1 - sensitivity`), precision, F1, AUROC, AUPRC, confusion matrix, and 95% patient/study-level bootstrap confidence intervals. This false-negative rate is not `1 - NPV`, because NPV depends on prevalence.

The target is a protocol goal, not a claim that a final advanced model has already achieved it on an untouched test set. Historical Phase-1 official-valid results remain reference evidence only; the governance limitation is documented in [`reports/TEST_SET_GOVERNANCE.md`](reports/TEST_SET_GOVERNANCE.md).

## 3. Data + Plan B

**Plan A — Stanford MURA v1.1, XR_WRIST MVP.** Data are organised as **patient -> study -> one or more views**; the task is study-level NORMAL / ABNORMAL classification. Official MURA `train` is split at patient level into internal train and validation. Logistic Regression regularization is chosen with patient-grouped CV inside train; each threshold is selected only from aggregated internal-validation study predictions.

Official MURA `valid` was used once for the frozen Phase-1 reference evaluation and its results were observed. It is therefore not an untouched population for future model, threshold, architecture, calibration, or agent selection. Future advanced-model claims require a transparent development-CV limitation, compatible external validation, or a lecturer-approved protocol; the team will not invent a new test set from already-used data.

MURA is handled under its applicable research/non-commercial terms. Raw images, credentials, and protected local manifests never enter Git.

**Plan B — FracAtlas.** Activating Plan B changes the scientific target from general MURA abnormality to fracture detection and changes labels, supported anatomy, QC assumptions, and evaluation design. It therefore requires an explicit documented adaptation, not a claim that the downstream workflow is identical.

## 4. Architecture sketch

```text
Data -> patient / study / views -> image model -> mean study-level aggregation
     -> versioned probability + evidence/disagreement
     -> deterministic safety and routing policy (works without an agent)
     -> routine human workflow  OR  ambiguity investigation
                                      -> bounded adaptive agent
                                      -> structured evidence-grounded handoff
                                      -> human clinician
```

The mandatory baseline is **frozen ImageNet ResNet-18 embeddings -> mean study embedding -> Logistic Regression**, compared fairly with fine-tuned ResNet-18 on the same population and metrics. Mean aggregation is primary; other aggregation is secondary unless pre-registered.

Hard rules are deterministic: supported-anatomy checks, corrupt/missing-input handling, required human review, and the frozen operating policy. The AI may raise a case's priority for human review; it must never reduce a case below standard review or override a guardrail.

## 5. Agent seat

**Charter status: adaptive agent designed, not evaluated.** A fixed script handles known rules and routine cases. The current Wrist Review product separately implements deterministic routing and an opt-in bounded CrewAI research-explanation bridge; it is not presented as an evaluated adaptive planner. A bounded adaptive agent remains proposed for supported cases with case-specific ambiguity, for example:

1. view-level evidence conflicts substantially;
2. the fixed decision is borderline and QC evidence is also borderline; or
3. an allowed tool returns missing or conflicting evidence.

The agent may identify missing approved evidence, request one permitted uncertainty/TTA check, and write a structured provenance-grounded handoff. Its allowlist is input QC, support check, versioned primary model, per-view evidence/disagreement, approved uncertainty tools, Grad-CAM as a shortcut diagnostic only, model card, and calculator. It may return `NEEDS_HUMAN_REVIEW` or `UNSUPPORTED`; it may not diagnose, treat, alter thresholds, bypass guardrails, or claim localization.

On timeout, malformed output, or unresolved conflict, the deterministic route remains and the case goes to human review. The same independently authored cases will compare (A) deterministic routing, (B) fixed workflow, and (C) adaptive agent for escalation/rejection, groundedness, evidence, tool choice, unnecessary calls, and guardrail violations. Metadata/document text is untrusted; embedded instructions cannot change policy.

## 6. Risks + cut line

| Risk | Mitigation |
|---|---|
| MURA abnormality is not acute fracture | Maintain honest scope; do not make fracture claims; use defensible error review only. |
| Shortcut learning from markers, text, casts, metal, borders, or acquisition artifacts | Use targeted error/shortcut analysis; Grad-CAM is a coarse diagnostic aid, not localization ground truth. |
| Observed official-valid results and limited sample size can overstate certainty | Keep official-valid out of all later selection; report bootstrap CIs and the reuse limitation. |
| Agent becomes an if/else wrapper or hallucinates | Deterministic non-overridable guardrails, typed tools, hostile-input tests, and independent agent evaluation. |

**Cut line:** a reproducible study-level wrist-abnormality classifier with the mandatory baseline, patient-safe protocol, high-sensitivity operating point, bootstrap/error/shortcut analysis, deterministic safety, and separately evaluated bounded agent design. Stronger models, learned aggregation, calibration, Grad-CAM, and a UI are optional.

## 7. Milestones + hats

| Checkpoint | Deliverable | Owner / hat |
|---|---|---|
| Checkpoint 1 | Thin end-to-end baseline; patient-safe split, aggregation, validation-only threshold, baseline comparison | Hagit — Data; Hodaya — Core Model; Netanel — model-runtime integration, results interface, and separately labelled all-anatomy experimental extension |
| Checkpoint 2 | Agent integration and dedicated safety/evaluation suite; deterministic workflow remains operational | Netanel — Agent, deterministic safety, and local application integration |
| Final | Error/shortcut analysis, governance, rule-vs-agent comparison, demo/docs, future-cost estimate | Artium — Product; Netanel — application/demo integration; Hagit — data/scope review; Hodaya — model review |

### Current workstream ownership

| Workstream | Owner(s) | Scope of responsibility |
|---|---|---|
| Data | Hagit | Dataset access, patient/study/view structure, split and preprocessing governance. |
| Core ML | Hodaya | Fine-tuned model workflow and model-training evidence. |
| Model integration and experimental extension | Netanel | Canonical runtime/model-bundle integration, study-level result presentation, model-artifact governance, and coordination of the separately labelled experimental all-anatomy extension. |
| Agent and deterministic safety | Netanel | Bounded-agent integration, deterministic routing/safety policy, independent agent evaluation, and hostile-input safeguards. |
| Application and demo | Netanel | Wrist Review local interface, safe model-to-UI integration, reviewer flow, and final demo integration. |
| Product | Artium | Product framing, workflow/reproducibility review, and contribution to final presentation readiness. |

Named hats describe current ownership and coordination, not a claim that one
person individually implemented every historical notebook or future component.
The all-anatomy workstream remains experimental and is not part of the
validated XR_WRIST MVP claim. Source/citation items that require verification
are tracked in [`reports/SOURCE_VERIFICATION.md`](reports/SOURCE_VERIFICATION.md).
