# Deterministic policy — proposal for team review (rev. 2)

**Status:** proposal (28.10.2026, rev. 2), not implemented, not yet agreed by the team. Independent of the Checkpoint 1 agent stub. Rev. 2 aligns the proposal with `agents/AGENT_DESIGN.md` without changing that document (change log in §7).

**Basis:** `CHARTER.md` §4–5, `agents/AGENT_DESIGN.md` §1–7, `evaluation/AGENT_EVAL_PLAN.md` (main @ 77f6ebc).

## 1. Outcome vocabulary (v1)

| Outcome | When | Issued by |
|---|---|---|
| `REJECTED` | processing is technically impossible, or patient identity is missing/inconsistent | deterministic layer only |
| `UNSUPPORTED` | anatomy is not wrist | deterministic layer; the agent only when a tool result confirms it (AGENT_DESIGN §2, §4) |
| `NEEDS_HUMAN_REVIEW` | malformed study; tool failure; hostile input; escalation by the agent | deterministic layer or agent |
| `NOT_FLAGGED` | study probability below the frozen threshold (not a "normal" claim) | deterministic layer |
| `FLAGGED_REVIEW` | study probability ≥ threshold: abnormality cannot be excluded by the model | deterministic layer |

- A separate field `model_result` (`p`, `threshold`, `below_threshold` / `above_threshold`) is filled only when the model ran.
- `NOT_FLAGGED` / `FLAGGED_REVIEW` are the "deterministic route" in AGENT_DESIGN terms.
- **The agent returns one of three values** (AGENT_DESIGN §2): `NO_CHANGE` (the deterministic route is kept; the agent adds a grounded summary), `NEEDS_HUMAN_REVIEW`, or `UNSUPPORTED` (only when backed by a `support_check` result; otherwise it becomes `NEEDS_HUMAN_REVIEW` and is logged as a policy violation).
- Final outcome = deterministic route if the agent was not called or returned `NO_CHANGE`; otherwise the agent's outcome. The agent cannot lower an outcome: no such value exists in its output schema.

## 2. Input and QC rules (precedence as in AGENT_DESIGN §1)

**Principle:** reject only when processing is technically impossible or identity is broken. A readable but questionable image is never rejected.

| §1 row | Condition | Outcome | Model runs |
|---|---|---|---|
| 1 | file cannot be decoded / unsupported format | `REJECTED` | no |
| 1 | blank or near-uniform image | `REJECTED` | no |
| 1 | missing patient ID, or different IDs in one study | `REJECTED` | no |
| 1 | zero images in the study | `REJECTED` | no |
| 1 | different body parts in one study (malformed study) | `NEEDS_HUMAN_REVIEW` | no |
| 1 | more than N images (malformed study; N from official train) | `NEEDS_HUMAN_REVIEW` | no |
| 2 | anatomy is not wrist | `UNSUPPORTED` | no |
| 3 | metadata outside the controlled vocabulary (hostile or invalid input, AGENT_DESIGN §6–7) | `NEEDS_HUMAN_REVIEW` | no |
| 3 | failure of a key tool; malformed or missing tool output (AGENT_DESIGN §4, §7) | `NEEDS_HUMAN_REVIEW` | — |
| — | byte-identical duplicate image (sha256) | remove duplicate, continue, record in evidence — the only allowed change of study membership | yes |
| — | soft QC flag: over/under-exposure, blur, crop, rotation (only if measurable) | continue; flag recorded in evidence; triggers the agent | yes |

- AGENT_DESIGN §1 row 3 ("mandatory safety or human-review rule") is defined here as the two rows marked 3. In v1 the agent is not called for them (§1 allows it there only for a non-blocking summary).
- Non-standard projection is out of QC scope: it cannot be detected deterministically (MURA has no view labels).

## 3. Agent triggers

The agent runs only on supported, valid studies when at least one trigger fires.

| # | Trigger | Side of threshold |
|---|---|---|
| 1 | strong disagreement between per-view probabilities | any |
| 2 | soft QC flag (on its own) | any |
| 3 | borderline result: threshold − δ ≤ p < threshold | below only |

- The rev. 1 trigger "partial missing/conflicting evidence" is removed: per AGENT_DESIGN §7, missing or malformed tool output gives deterministic `NEEDS_HUMAN_REVIEW` (§2 above).
- Trigger 2 extends the examples in CHARTER §5 (the list there is "for example").
- Trigger 3 is below-threshold only: the agent may act only upward, so only below the threshold can it correct an error (a possible false negative). A missing flag may also reduce reviewer attention (automation bias; Goddard et al., JAMIA 2012;19(1):121–127). The benefit is a hypothesis to be tested on the agent evaluation suite.

## 4. Agent failures and budget (AGENT_DESIGN §7)

- Timeout, invalid LLM output, conflicting evidence without a permitted resolution, or hostile input → the deterministic route is kept in the record and the final outcome is `NEEDS_HUMAN_REVIEW`. No LLM retry in v1.
- Budget: at most 6 tool calls per case, **including** pipeline calls (`input_qc`, `support_check`, `primary_model`, `view_evidence`); at most one uncertainty/TTA call. Each call logs latency, CPU/GPU time, API tokens and estimated cost under a versioned price assumption.

## 5. Meaning of the flag (Model B, internal validation)

Threshold 0.2636 (Se ≥ 95%). Flagged: 62.5% of studies, PPV 0.593. Not flagged: NPV 0.950. PPV/NPV depend on prevalence (39% in internal validation). Wording for the clinician: "abnormality cannot be excluded by the model (MURA-defined); careful review required".

## 6. Open parameters

| Parameter | Depends on |
|---|---|
| disagreement threshold (trigger 1) | per-view predictions of the frozen Model B |
| δ (trigger 3); number of the 13 FN inside the δ band | study predictions of the frozen Model B |
| share of studies routed to the agent | both above |
| N (max images per study) | official train only |
| soft QC thresholds and their firing rate | official train only |
| deterministic rotation check | method not chosen |
| reading of the §7 budget ("per case" includes pipeline calls) | team decision |

All numeric parameters are fixed once, before the agent evaluation suite is run. Official MURA valid is not used for any of them.

## 7. Change log (rev. 1 → rev. 2)

| Rev. 1 | Rev. 2 | AGENT_DESIGN |
|---|---|---|
| agent proposes `NOT_FLAGGED` / `FLAGGED_REVIEW` / `NEEDS_HUMAN_REVIEW`; final = max | agent returns `NO_CHANGE` / `NEEDS_HUMAN_REVIEW` / `UNSUPPORTED` (tool-backed) | §2 |
| `UNSUPPORTED` from the deterministic layer only | also from the agent, only when tool-backed | §2 |
| partial missing evidence → agent (trigger 3) | → deterministic `NEEDS_HUMAN_REVIEW`; 3 triggers remain | §4, §7 |
| LLM failure → fall back to the deterministic route | → `NEEDS_HUMAN_REVIEW` | §7 |
| hostile input not covered | metadata outside the vocabulary → `NEEDS_HUMAN_REVIEW` | §6, §7 |
| mixed study / > N images as separate rules | classified as §1 row 1 ("malformed study") | §1 |
| "≤ 6 steps" | ≤ 6 tool calls per case, including the pipeline | §7 |

No change lowers a case below standard review; the cost is a higher share of `NEEDS_HUMAN_REVIEW`.

## 8. Risks

- Agent load: trigger 1 alone may cover about a third of studies (high-disagreement group, Phase 2C subgroup report); triggers 2 and 3 add more.
- `NEEDS_HUMAN_REVIEW` share grows (agent escalations, tool and LLM failures); reasons are distinguishable only by reason codes, which must be shown to the clinician.
- Trigger 3 benefit is unproven if false negatives lie far from the threshold.
- Soft QC thresholds have no ground truth in MURA; firing on noise must be measured.
- A 6-call budget that includes pipeline calls leaves the agent ≤ 2 calls; this may be too few once TTA and Grad-CAM are added.
