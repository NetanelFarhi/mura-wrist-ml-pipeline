# Requirements traceability

Status is intentionally conservative: **DONE**, **IN PROGRESS**, **PLANNED**, **BLOCKED**, or **NEEDS DECISION**. “DONE” describes an existing repository artifact, not a claim of clinical readiness.

## BIU DS23 charter and repository requirements

| Requirement | Evidence / planned artifact | Status | Deadline / checkpoint |
|---|---|---|---|
| 1. Quantified problem + specific user | `CHARTER.md` §1 | DONE | 4.10.2026 |
| 2. Numeric success metric and pre-build evaluation plan | `CHARTER.md` §2; `reports/TEST_SET_GOVERNANCE.md` | DONE | 4.10.2026 |
| 3. Data source/size/license/access and one-line Plan B | `CHARTER.md` §3; `data/README.md`; `reports/wrist_data_audit.json` | IN PROGRESS | 4.10.2026 — official license citation still needed |
| 4. Architecture sketch | `CHARTER.md` §4 | DONE | 4.10.2026 |
| 5. Unique agent seat, tools, failure behavior, works without agent | `CHARTER.md` §5; `agents/AGENT_DESIGN.md` | DONE (design) | 4.10.2026 |
| 6. Three risks + cut line | `CHARTER.md` §6 | DONE | 4.10.2026 |
| 7. Milestones + named Data/Model/Agent/Product hats | `CHARTER.md` §7 | BLOCKED | 4.10.2026 — team names required |
| Root `CHARTER.md`, max two pages | `CHARTER.md`; `reports/SUBMISSION_2026-10-04.md` | IN PROGRESS | 4.10.2026 — render check required |
| README, reproducibility, data dictionary, no raw data | `README.md`; `data/data_dictionary.md`; `data/README.md`; `.gitignore` | DONE | Checkpoint 1 |
| Cheap baseline comparison | `README.md`; `reports/results.md`; `reports/baseline_results.md` | DONE | Checkpoint 1 |
| Thin end-to-end baseline without final agent | existing Phase-1 pipeline and reports | DONE | 18.10.2026 |
| Agent integrated and demo-ready | `agents/AGENT_DESIGN.md`; implementation pending | PLANNED | 30.10.2026 |
| Final evaluation/demo/docs/cost | `evaluation/`; `reports/PRODUCT_AND_COST_PLAN.md` | PLANNED | 12.11.2026 |

## Lecturer Part A — before charter submission

| Correction | Evidence / planned artifact | Status | Deadline |
|---|---|---|---|
| Measurable sensitivity-constrained success metric; no revisit outcome/ambiguous “3% misses” | `CHARTER.md` §2; `reports/SOURCE_VERIFICATION.md` | DONE | 4.10.2026 |
| Patient-safe internal validation; threshold never selected on official-valid test | `CHARTER.md` §3; `reports/TEST_SET_GOVERNANCE.md`; Phase-1 reports | DONE | 4.10.2026 |
| Justified two-layer agent: deterministic safety plus adaptive ambiguity only | `CHARTER.md` §§4–5; `agents/AGENT_DESIGN.md` | DONE (design) | 4.10.2026 |
| Rule/fixed-workflow/agent comparison on same independent cases | `evaluation/AGENT_EVAL_PLAN.md` | PLANNED | 30.10.2026 |
| Mandatory frozen-feature Logistic Regression baseline | `README.md`; `reports/results.md` | DONE | 4.10.2026 |

## Lecturer Part B — development alignment

| Requirement | Evidence / planned artifact | Status | Deadline |
|---|---|---|---|
| Early thin baseline, no final agent required | Phase-1 pipeline; `CHARTER.md` §7 | DONE | 18.10.2026 |
| Plan B changes task/anatomy/QC/evaluation assumptions | `CHARTER.md` §3 | DONE | 4.10.2026 |
| Research/non-commercial licensing; no raw images in Git | `data/README.md`; `.gitignore`; `scripts/check_repo_data_governance.py` | IN PROGRESS | 4.10.2026 — verify official terms |
| Product hat and named ownership | `CHARTER.md` §7 | BLOCKED | 4.10.2026 — team names required |
| Future operating-cost estimate | `reports/PRODUCT_AND_COST_PLAN.md` | PLANNED | 12.11.2026 |
| Hostile-input/prompt-injection guardrails | `agents/AGENT_DESIGN.md`; `evaluation/AGENT_EVAL_PLAN.md` | PLANNED | 30.10.2026 |
| Explicit study aggregation | `CHARTER.md` §4; `README.md` Models section | DONE | 4.10.2026 |

## Lecturer Part C — analytical and safety requirements

| Requirement | Evidence / planned artifact | Status | Deadline |
|---|---|---|---|
| Scope MURA abnormality, not acute fracture | `CHARTER.md` §1; `README.md`; `data/README.md` | DONE | 4.10.2026 |
| Error/shortcut analysis with no fabricated subtypes | `evaluation/SHORTCUT_ANALYSIS_PLAN.md` | PLANNED | 12.11.2026 |
| Grad-CAM as diagnostic only; no lesion-localization claim | `evaluation/SHORTCUT_ANALYSIS_PLAN.md`; agent design | PLANNED | 12.11.2026 |
| Bootstrap confidence intervals / no over-precision | `CHARTER.md` §2; Phase-1/2 reports | IN PROGRESS | 12.11.2026 |
| Safety asymmetry: raise priority only | `CHARTER.md` §4; `agents/AGENT_DESIGN.md` | DONE (design) | 4.10.2026 |
| Independent agent safety suite, not circular threshold ground truth | `evaluation/AGENT_EVAL_PLAN.md` | PLANNED | 30.10.2026 |
| Published MURA wrist-performance context with non-identical-protocol disclaimer | `reports/SOURCE_VERIFICATION.md` | NEEDS DECISION | 12.11.2026 |
| Verify/remove Graif/revisit claim | `reports/SOURCE_VERIFICATION.md` | IN PROGRESS | 4.10.2026 |
| Honest governance after official-valid was observed | `reports/TEST_SET_GOVERNANCE.md` | DONE | 4.10.2026 |

## Open decisions requiring team or lecturer input

1. Replace every `[NAME REQUIRED]` in the Charter with verified team members and the four hat owners.
2. Choose the approved advanced-model evaluation protocol: transparent development CV, compatible external validation, or lecturer-approved alternative. Do not invent a new untouched test from used data.
3. Verify the official MURA license/source terms and any clinical background citation before final wording.
