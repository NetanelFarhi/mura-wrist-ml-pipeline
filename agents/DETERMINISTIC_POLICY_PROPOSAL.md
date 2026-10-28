# Deterministic policy — proposal for team review

**Status:** proposal (28.10.2026), not implemented, not yet agreed by the team. Independent of the Checkpoint 1 agent stub.
**Basis:** `CHARTER.md` §4–5, `agents/AGENT_DESIGN.md` §1, §4–7, `evaluation/AGENT_EVAL_PLAN.md` (main @ 77f6ebc).

## 1. Outcome vocabulary (v1)

| Outcome | When |
|---|---|
| `REJECTED` | processing is technically impossible, or patient identity is missing/inconsistent |
| `UNSUPPORTED` | anatomy is not wrist |
| `NEEDS_HUMAN_REVIEW` | the study cannot be scored and needs a human; or escalation by the agent |
| `NOT_FLAGGED` | study probability below the frozen threshold (not a "normal" claim) |
| `FLAGGED_REVIEW` | study probability ≥ threshold: abnormality cannot be excluded by the model |

- A separate field `model_result` (`p`, `threshold`, `below_threshold` / `above_threshold`) is filled only when the model ran.
- Order ("only upward"): `NOT_FLAGGED` < `FLAGGED_REVIEW` < `NEEDS_HUMAN_REVIEW`. Final outcome = max(deterministic outcome, agent proposal). Any downgrade attempt is ignored and logged as a policy violation.
- `REJECTED` and `UNSUPPORTED` are issued only by the deterministic layer; such cases never reach the agent.

## 2. Input and QC rules

Principle: reject only when processing is technically impossible or identity is broken. A readable but questionable image is never rejected.

| Condition | Outcome | Model runs |
|---|---|---|
| file cannot be decoded / unsupported format | `REJECTED` | no |
| blank or near-uniform image | `REJECTED` | no |
| missing patient ID, or different IDs in one study | `REJECTED` | no |
| zero images in the study | `REJECTED` | no |
| anatomy is not wrist | `UNSUPPORTED` | no |
| different body parts in one study | `NEEDS_HUMAN_REVIEW` | no |
| more than N images (N from official train distribution) | `NEEDS_HUMAN_REVIEW` | no |
| byte-identical duplicate image (sha256) | remove duplicate, continue, record in evidence — the only allowed change of study membership | yes |
| soft QC flag: over/under-exposure, blur, crop, rotation (only if measurable) | continue; flag recorded in evidence; triggers the agent | yes |

Non-standard projection is out of QC scope: it cannot be detected deterministically (MURA has no view labels).

## 3. Agent triggers

The agent runs only on supported, valid studies when at least one trigger fires.

| # | Trigger | Side of threshold |
|---|---|---|
| 1 | strong disagreement between per-view probabilities | any |
| 2 | soft QC flag (on its own) | any |
| 3 | partial missing/conflicting evidence while the model returned a result | any |
| 4 | borderline result: threshold − δ ≤ p < threshold | below only |

- Complete failure of a key tool (model or QC) → deterministic `NEEDS_HUMAN_REVIEW` with reason; the agent is not called.
- Trigger 2 extends the examples in CHARTER §5 (the list there is "for example").
- Trigger 4 is below-threshold only: the agent may act only upward, so only below the threshold can it correct an error (a possible false negative). A missing flag may also reduce reviewer attention (automation bias; Goddard et al., JAMIA 2012;19(1):121–127). The benefit is a hypothesis to be tested on the agent evaluation suite.

## 4. Meaning of the flag (Model B, internal validation)

Threshold 0.2636 (Se ≥ 95%). Flagged: 62.5% of studies, PPV 0.593. Not flagged: NPV 0.950. PPV/NPV depend on prevalence (39% in internal validation). Wording for the clinician: "abnormality cannot be excluded by the model (MURA-defined); careful review required".

## 5. Open parameters

| Parameter | Depends on |
|---|---|
| disagreement threshold (trigger 1) | per-view predictions of the frozen Model B |
| δ (trigger 4); number of the 13 FN inside the δ band | study predictions of the frozen Model B |
| share of studies routed to the agent | both above |
| N (max images per study) | official train only |
| soft QC thresholds and their firing rate | official train only |
| content of the "mandatory human-review rule" (AGENT_DESIGN §1, row 3) | team decision |

All numeric parameters are fixed once, before the agent evaluation suite is run. Official MURA `valid` is not used for any of them.

## 6. Risks

- Agent load: trigger 1 alone may cover about a third of studies (high-disagreement group, Phase 2C subgroup report); triggers 2 and 4 add more.
- Trigger 4 benefit is unproven if false negatives lie far from the threshold.
- Soft QC thresholds have no ground truth in MURA; firing on noise must be measured.
