# MURA XR_WRIST Research Project

**BIU DS23 final project**

**Team:** Netanel, Hagit, Hodaya, Artium

## Current team workstreams

This table records the team's current ownership and coordination areas. It
does not rewrite historical notebook authorship or claim that planned work is
already complete.

| Team member | Current ownership |
|---|---|
| **Netanel** | Wrist Review application and demo integration; canonical model-runtime/artifact integration; study-level result presentation; deterministic safety and bounded-agent integration/evaluation; coordination of the separately labelled experimental all-anatomy extension. |
| **Hagit** | Data access and governance; patient/study/view organisation; preprocessing and data-scope review. Agent design |
| **Hodaya** | Core fine-tuned model workflow and model-training evidence. |
| **Artium** | Product framing, reproducibility review, and final-presentation readiness. |

The all-anatomy extension remains **experimental** and is not a validated
replacement for the XR_WRIST MVP. The adaptive agent remains **planned** until
it is implemented and independently evaluated.

Course project for study-level **NORMAL / ABNORMAL** classification of wrist
radiograph studies from Stanford MURA v1.1. This is a research prototype for
decision support before specialist review. It predicts the MURA study label;
it is **not** a fracture diagnosis, fracture-localization system, or autonomous
clinical decision tool.

> **Reviewer start here:** [Current project state and navigation guide](docs/REVIEW_GUIDE.md).
> It distinguishes the current internal-validation evidence, the historical
> once-opened official MURA `valid` reference, experimental extensions, and
> planned Agentic-AI work.
> [Result lineage](docs/RESULT_LINEAGE.md) explains why historical 2,768/692
> and current 2,769/691 internal split results must not be mixed.

## Problem, user, and scope

The intended user is an **Emergency Department physician** reviewing a wrist
radiograph study before specialist review is available. The project studies
whether a study-level model can flag **MURA-defined musculoskeletal
abnormality** for priority human review. MURA `ABNORMAL` is not synonymous
with acute fracture; the project does not measure diagnosis time, return
visits, treatment benefit, lesion location, or clinical deployment readiness.
Every case remains in human review.

## Current notebook workflow

The numbered notebooks below are the current Checkpoint 1 workflow. Run them
in order, using the same patient-safe internal train/validation split.

| # | Notebook | Purpose | Status |
|---:|---|---|---|
| 01 | [Data preparation and EDA](notebooks/01_MURA_Data_Preparation_and_EDA.ipynb) | Creates the wrist-only manifest, documents the study/view structure, and makes an internal patient-level split from official MURA train. | Current |
| 02 | [Preprocessing](notebooks/02_MURA_Preprocessing.ipynb) | Validates and cleans manifests, verifies image decoding, and defines deterministic evaluation preprocessing. | Current |
| 03 | [Frozen ResNet-18 baseline](notebooks/03_MURA_Baseline_ResNet18_LogReg.ipynb) | Extracts frozen ImageNet ResNet-18 embeddings, mean-pools views within a study, and fits Logistic Regression. | Current baseline |
| 04 | [Baseline evaluation](notebooks/04_MURA_Baseline_Evaluation.ipynb) | Evaluates the baseline on internal validation, including ROC/PR, threshold policy, bootstrap intervals, and error tables. | Current baseline evaluation |
| 05 | [Fine-tuned ResNet-18](notebooks/05_MURA_FineTuned_ResNet18.ipynb) | Defines the controlled study-level Model B experiment using the same split as the baseline. | Completed on internal validation |
| 06 | [Model B evaluation and comparison](notebooks/06_MURA_ModelB_Evaluation_and_Comparison.ipynb) | Compares Model A and Model B on the same internal validation studies and reports a validation-only operating policy. | Completed on internal validation |
| 07 | [Shortcut and geometry audit](notebooks/07_MURA_Shortcut_and_Geometry_Audit.ipynb) | Tests geometry/padding sensitivity and records diagnostic evidence about possible shortcut learning. | Completed diagnostic audit |
| 08 | [Local runtime study viewer](notebooks/08_MURA_Runtime_Study_Viewer.ipynb) | Displays the separate views of one authorised local internal-train study for visual QC. It is code-only in Git and defaults to no displayed image output. | Optional local/Colab utility |
| 09a | [Deterministic safety tools](notebooks/09_MURA_Deterministic_Safety_Tools.ipynb) | Implements and tests input-QC, scope validation, disagreement evidence, and non-overridable safe routing. | Current deterministic safety baseline |
| 09b | Agentic AI — CrewAI Research Explanation Agent | Implements an experimental single-agent CrewAI workflow using Gemini 3.1 Flash-Lite, deterministic policy-gated tool selection, TTA and Grad-CAM explanations, and execution cost reporting. | Implemented and successfully demonstrated — experimental contributor workflow; not yet integrated into the canonical application |

Earlier notebooks are preserved under [notebooks/legacy](notebooks/legacy/)
for traceability. They are not the canonical workflow and must not be used to
make new selection decisions without protocol review.

The [experimental all-anatomy extension](notebooks/experimental_all_anatomies/)
contains separately labelled development notebooks for MURA's seven anatomy
categories. It is not part of the XR_WRIST MVP claim and has not been promoted
to a final multi-anatomy result.

The canonical notebooks retain reviewed aggregate tables and figures so their
results render directly on GitHub. Radiographs, study IDs, Drive paths,
checkpoints, embeddings, and detailed prediction files must remain outside Git.

## Checkpoint 1 results — current development protocol

The current comparison uses the same internal, patient-level validation split:
691 studies from 653 patients (422 NORMAL, 269 ABNORMAL), drawn from official
MURA train. It is a **development/validation** analysis, not a final clinical
claim or an untouched-test result.

- **Model A baseline:** frozen ImageNet ResNet-18 embeddings, mean embedding
  pooling at study level, and Logistic Regression. Its regularization value
  (`C`) is selected only with five-fold patient-grouped cross-validation inside
  the internal training split; the selected model is then fit on all internal
  training studies.
- **Model B primary model:** the existing fine-tuned ResNet-18 is reported
  under the revised protocol's primary mean probability pooling policy at
  study level. Max probability pooling is retained only as a secondary
  diagnostic comparison; Model B was not retrained for this reporting update.
- **Operating policy:** for each model independently, the threshold is the
  highest threshold that reaches **internal-validation sensitivity >=95%** on
  study-level probabilities. The threshold is frozen from that validation
  analysis; it is not transferred from one model to another.

| Same 691-study internal-validation cohort | AUROC | AUPRC | Threshold | Sensitivity | Specificity | FP / FN | Flagged |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen ResNet-18 embeddings + Logistic Regression | 0.880 | 0.852 | 0.063774 | 95.2% | 39.3% | 256 / 13 | 74.1% |
| Fine-tuned ResNet-18, mean probability pooling | 0.913 | 0.878 | 0.263568 | 95.2% | 58.3% | 176 / 13 | 62.5% |

This is the mandatory cheap-baseline comparison on the same development
population and operating target. Comparable training/inference time and cost
were not saved for these two runs, so no speed or cost superiority is claimed.
Fine-tuned max pooling is secondary diagnostic evidence: it reached 62.1%
specificity at 95.2% sensitivity but lower AUROC/AUPRC (0.883 / 0.806).

The generated [current Checkpoint 1 results page](docs/results/checkpoint1/README.md)
contains the exact AUROC, AUPRC, threshold, sensitivity, specificity, review
workload, false-positive/false-negative counts, and patient-cluster bootstrap
intervals. It is the source of the currently reported numbers rather than this
README. If an approved protocol-preserving rerun updates that result page, the
summary table in this README must be updated in the same reviewed change.

The prior >=90% development-policy outputs are preserved as historical,
pre-feedback material under
[docs/results/archive/checkpoint1_pre_95pct](docs/results/archive/checkpoint1_pre_95pct/).
They are not the current operating-policy result.

## Current controlled multi-view result — development only

One pre-registered Phase 2C experiment tested whether unfreezing only the
highest ResNet-18 block improves a study-level **mean embedding** model. It
did not: the frozen reference was stronger on internal validation (AUROC
0.914983 / AUPRC 0.878652 versus 0.913767 / 0.873423), and at the
pre-specified >=95% sensitivity policy it retained higher specificity (0.585308
versus 0.571090). Patient-cluster bootstrap intervals for all paired differences
included zero. The project therefore freezes the simpler frozen mean-embedding
ML core rather than continuing architecture tuning.

The aggregate tables, paired uncertainty, training curve, error counts and
diagnostic checks are in [reports/phase2c/PHASE2C_SUMMARY.md](reports/phase2c/PHASE2C_SUMMARY.md).
The rendered [results viewer](notebooks/06_results_viewer.ipynb) contains those
safe aggregate outputs for GitHub. It contains no raw radiographs or per-study
predictions.

**Important model boundary:** the Phase 2C frozen mean-embedding reference is
a separate research comparison. The current Checkpoint 1 reporting and local
deterministic app retain the fine-tuned Model B **mean-probability** policy and
its model-specific threshold. The two models and thresholds must never be
interchanged; see [result lineage](docs/RESULT_LINEAGE.md).

## Agentic AI and Safety Status

The MURA XR_WRIST project combines a deterministic safety framework with an experimental Agentic AI workflow designed to support research-level interpretation of model predictions.

### 1. Deterministic Safety Layer — Implemented

The deterministic safety layer performs input quality control, anatomical scope validation, view-consistency checks, and controlled routing to human review.

The system preserves the frozen ML classification and its selected threshold. The safety policy cannot be overridden by the language model.

All cases remain subject to human review.

### 2. Agentic AI — Experimental Implementation

An experimental Agentic AI workflow has been implemented in the contributor notebook `09_MURA_Agentic_AI_semi_final(2).ipynb`.

The implementation uses **CrewAI with Gemini 3.1 Flash-Lite** and follows a single-agent architecture.

**Agent: MURA Research Explanation Analyst**

The agent is responsible for interpreting validated research evidence, invoking approved explanation tools, and generating a structured research report.

Two supplementary explanation tools are available:

- **Test-Time Augmentation (TTA):** Provides evidence about prediction stability under image transformations.
- **Grad-CAM:** Provides spatial explanation evidence to support interpretation of the model's behavior.

Tool selection is controlled by a deterministic policy (`tools/agent_policy.py`, version 1.1). The policy validates workflow consistency and determines which optional research tools are permitted.

The agent cannot modify model predictions, classification thresholds, or mandatory human-review requirements.

### 3. Experimental Execution — Completed

A documented experimental execution was successfully completed using a selected XR_WRIST study.

During this execution:

- The CrewAI agent invoked both approved explanation tools.
- The tools returned successful results.
- The agent generated a research explanation report.
- The original ML classification remained unchanged.
- LLM usage, execution time, and estimated API costs were recorded.

The execution demonstrated technical feasibility of the policy-gated explanation workflow.

### 4. Current Limitations and Next Steps

The Agentic AI implementation remains an **experimental contributor workflow** and has not yet been integrated into the canonical Wrist Review application.

Independent agent evaluation, systematic testing across prediction categories, reproducibility checks, and cost-control mechanisms remain future work.

The system is intended exclusively for research and educational purposes. It is not validated for autonomous clinical decision-making, fracture diagnosis, or clinical deployment.

## Data and evaluation governance

Raw MURA images, ZIP archives, per-study predictions, embeddings, checkpoints,
credentials, and local Drive paths must never be committed. See
[data/README.md](data/README.md), [data/data_dictionary.md](data/data_dictionary.md),
and [.gitignore](.gitignore) for the expected local data layout and safeguards.

All views from a study remain together, and a patient may not appear in both
internal train and validation. The official MURA `valid` split was used once
for the historical Phase 1 frozen-reference evaluation. Because that result has
already been observed, it is not an untouched test set for future architecture,
threshold, calibration, or agent decisions. It is not read for the current
Checkpoint 1 model/threshold-selection workflow. See
[reports/TEST_SET_GOVERNANCE.md](reports/TEST_SET_GOVERNANCE.md).

## Scope and limitations

- MURA `ABNORMAL` is not synonymous with acute fracture.
- A high-sensitivity threshold increases review workload; it is a research
  tradeoff, not a clinical policy.
- Model scores are not automatically calibrated confidence.
- Geometry/shortcut checks are diagnostic evidence only. They cannot prove
  anatomical reasoning or lesion localization.
- No model in this repository is ready for clinical use.

## Repository map

| Location | Purpose |
|---|---|
| `data/` | Local-access instructions, schemas, and data dictionary; never raw MURA images. |
| `notebooks/` | Ordered wrist workflow; `legacy/` and `experimental_all_anatomies/` are explicitly non-canonical. |
| `src/`, `configs/`, `models/` | Reusable code, versioned settings, and safe model metadata; weights/predictions remain local. |
| `agents/`, `evaluation/` | Agent design, guardrails, and independent evaluation plans. |
| `docs/`, `reports/` | Current results, lineage, governance, product plans, and reviewer material. |
| `apps/wrist_review/` | Local research-interface source only; no model bundle or patient data. |
| `tests/` | Software, integrity, and safety tests; not ML-performance evidence. |

## Local quick start

### Safe repository checks (no MURA data required)

Use Python 3.10 or 3.11 from the repository root:

```powershell
git clone https://github.com/NetanelFarhi/mura-wrist-ml-pipeline.git
cd mura-wrist-ml-pipeline
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest tests -q
python scripts/check_repo_data_governance.py
```

### Authorized local-data reproduction

Raw MURA data, manifests, checkpoints, embeddings, and detailed predictions
are deliberately outside Git. First read [data/README.md](data/README.md) and
obtain MURA through its authorized research/non-commercial route. The
development-only runner accepts local artifacts outside the repository and is
configured to avoid official MURA `valid`:

```powershell
python scripts/execute_canonical_notebooks.py `
  --kernel <registered-jupyter-kernel> `
  --jupyter-path <path-to-jupyter-kernel-resources> `
  --source-csv-root <authorized-MURA-root> `
  --mura-data-root <authorized-MURA-root> `
  --raw-zip <authorized-path-to-MURA-v1.1_files.zip> `
  --artifacts-root <outside-repository-artifact-directory>
```

This runner is a development reproduction tool, not a way to create a new
untouched test result. For the optional local Wrist Review interface, see
[apps/wrist_review/README.md](apps/wrist_review/README.md).

## Product and future-cost status

No production monthly price is claimed because volume, infrastructure, data
retention, integration, and any future API/model choice are unknown. The
[product and cost plan](reports/PRODUCT_AND_COST_PLAN.md) records the required
assumptions and cost categories: compute, storage, integration/security,
optional LLM/API, monitoring, and human operations.

## Project documentation

- [Reviewer guide / current project state](docs/REVIEW_GUIDE.md)
- [Checkpoint 2 status](reports/CHECKPOINT2_STATUS.md)
- [Repository review — 2026-10-30](reports/REPOSITORY_REVIEW_20261030.md)
- [CHARTER.md](CHARTER.md)
- [Requirements traceability](REQUIREMENTS_TRACEABILITY.md)
- [Product and cost plan](reports/PRODUCT_AND_COST_PLAN.md)
- [Shortcut-analysis plan](evaluation/SHORTCUT_ANALYSIS_PLAN.md)
- [Agent design](agents/AGENT_DESIGN.md)
- [Agent evaluation plan](evaluation/AGENT_EVAL_PLAN.md)
- [Wrist Review app overview](docs/app/Wrist_Review_App.md)
- [Presentation and demo preparation](presentation/README.md)
- [Historical Phase 1 summary](reports/PHASE1_SUMMARY.md)
