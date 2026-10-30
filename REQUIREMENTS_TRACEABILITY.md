# Requirements traceability

Status meanings: **DONE** = an auditable artifact exists; **IN PROGRESS** = work is underway but not yet verified end-to-end; **PLANNED** = designed but not implemented; **NEEDS DECISION** = team/lecturer choice required; **BLOCKED** = cannot proceed without an external input. A design document is not evidence of clinical or agent performance.

## BIU charter fields and lecturer Part A corrections

| Requirement | Repository evidence | Status | Checkpoint |
|---|---|---:|---|
| Specific user and scoped problem | `CHARTER.md` §1 | DONE | Charter |
| Measurable high-sensitivity target; no revisit outcome as success claim | `CHARTER.md` §2 | DONE | Charter |
| False-negative rate distinguished from NPV | `CHARTER.md` §2 | DONE | Charter |
| Patient-safe official-train -> internal train/val protocol | `notebooks/01_MURA_Data_Preparation_and_EDA.ipynb`; `reports/TEST_SET_GOVERNANCE.md` | DONE | Checkpoint 1 |
| Logistic Regression C selected inside train with patient-grouped CV | `src/models/frozen_embedding_baseline.py`; `scripts/train_logistic_baseline.py`; `notebooks/03_MURA_Baseline_ResNet18_LogReg.ipynb`; `docs/results/checkpoint1/model_a_grouped_cv_candidates.csv` | DONE | Checkpoint 1 |
| Model-specific threshold selected only on internal validation, at study level | `src/training/metrics.py`; notebooks 04/06; `docs/results/checkpoint1/frozen_operating_policies.json` | DONE | Checkpoint 1 |
| Honest official-valid governance after historical Phase-1 opening | `reports/TEST_SET_GOVERNANCE.md`; `CHARTER.md` §3 | DONE | Charter |
| Mandatory frozen-feature ResNet-18 + Logistic Regression baseline | `notebooks/03_MURA_Baseline_ResNet18_LogReg.ipynb`; `docs/results/checkpoint1/model_comparison.csv` | DONE (corrected development baseline) | Checkpoint 1 |
| Fine-tuned ResNet-18 comparison on same population/metrics | `notebooks/05_MURA_FineTuned_ResNet18.ipynb`; `notebooks/06_MURA_ModelB_Evaluation_and_Comparison.ipynb` | DONE (development evidence) | Checkpoint 1 |
| Explicit primary study aggregation | `CHARTER.md` §4; model/notebook protocol | DONE | Charter / Checkpoint 1 |
| Architecture and deterministic no-agent path | `CHARTER.md` §4 | DONE (design) | Charter |
| Adaptive agent justification beyond if/else routing | `CHARTER.md` §5; `agents/AGENT_DESIGN.md` | DONE (design) | Charter |
| Agent comparison on independent same-case suite | `evaluation/AGENT_EVAL_PLAN.md` | PLANNED | Checkpoint 2 |
| Agent implementation | no implementation artifact | PLANNED | Checkpoint 2 |
| Bootstrap CIs for performance estimates | `docs/results/checkpoint1/README.md`; generated safe result artifacts | DONE (internal-validation estimates) | Checkpoint 1 / final evaluation |

## Lecturer Part B: development and product alignment

| Requirement | Repository evidence | Status | Checkpoint |
|---|---|---:|---|
| Thin end-to-end baseline works before advanced agent | Phase-1 notebooks and historical reports | DONE | Checkpoint 1 |
| Plan B explicitly changes task, QC, anatomy, and evaluation assumptions | `CHARTER.md` §3 | DONE | Charter |
| Research/non-commercial data handling; no raw images in Git | `data/README.md`; `.gitignore`; governance checks | DONE for this branch; re-verify before every future push | Charter / every push |
| Named Data, Model, Agent, Product hats | `CHARTER.md` §7 | DONE | Charter |
| Product value/workflow and future operating cost | `reports/PRODUCT_AND_COST_PLAN.md`; `docs/app/Wrist_Review_App.md` | DONE (design/prototype documentation) | Final |
| Hostile metadata / prompt-injection guardrails | `agents/AGENT_DESIGN.md`; `evaluation/AGENT_EVAL_PLAN.md` | DONE (design; execution tests planned) | Checkpoint 2 |

## Lecturer Part C: analysis and safety

| Requirement | Repository evidence | Status | Checkpoint |
|---|---|---:|---|
| Do not equate MURA abnormality with acute fracture | `CHARTER.md` §1; `data/README.md` | DONE | Charter |
| Error / shortcut plan without fabricated subtypes | `evaluation/SHORTCUT_ANALYSIS_PLAN.md`; Notebook 07 diagnostic audit | DONE (plan plus diagnostic evidence) | Final |
| Grad-CAM is diagnostic only, never lesion-localization evidence | `CHARTER.md` §6; `evaluation/SHORTCUT_ANALYSIS_PLAN.md` | DONE (scope boundary; tool remains optional) | Final |
| Safety asymmetry: can raise priority, never reduce standard review | `CHARTER.md` §§4–5; `agents/AGENT_DESIGN.md` | DONE (design) | Charter |
| Independent agent safety evaluation; no circular threshold ground truth | `evaluation/AGENT_EVAL_PLAN.md` | PLANNED | Checkpoint 2 |
| Published MURA wrist-performance context with non-identical-protocol caveat | `reports/SOURCE_VERIFICATION.md` | NEEDS DECISION | Final |
| Clinical citations and Graif/revisit claim verified or removed | `reports/SOURCE_VERIFICATION.md` | IN PROGRESS | Charter / final |

## Open methodological decisions

1. For advanced-model claims, choose and pre-register one: patient-grouped development CV with transparent reuse limitation; compatible external validation; or a lecturer-approved final protocol. No artificial “new untouched test” may be created from development data.
2. Confirm current official MURA terms and source citations before final submission wording.
3. Decide whether compatible external data are available before making any performance-generalisation claim beyond the historical Phase-1 reference.
