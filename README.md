# MURA XR_WRIST Research Project

Course project for study-level **NORMAL / ABNORMAL** classification of wrist
radiograph studies from Stanford MURA v1.1. This is a research prototype for
decision support before specialist review. It predicts the MURA study label;
it is **not** a fracture diagnosis, fracture-localization system, or autonomous
clinical decision tool.

> **Reviewer start here:** [Current project state and navigation guide](docs/REVIEW_GUIDE.md).
> It distinguishes the current internal-validation evidence, the historical
> once-opened official MURA `valid` reference, experimental extensions, and
> planned Agentic-AI work.

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
| 09b | [Agentic-AI contributor notebook](notebooks/09_MURA_Agentic_AI.ipynb) | Preserved contributor implementation draft with its original narrative and legacy local artifact layout. | Historical/integration draft — not canonical runtime |

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

The generated [current Checkpoint 1 results page](docs/results/checkpoint1/README.md)
contains the exact AUROC, AUPRC, threshold, sensitivity, specificity, review
workload, false-positive/false-negative counts, and patient-cluster bootstrap
intervals. It is the source of the currently reported numbers rather than this
README, so the visible summary cannot become stale after a protocol-preserving
rerun.

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

## Project documentation

- [Reviewer guide / current project state](docs/REVIEW_GUIDE.md)
- [Repository review — 2026-10-30](reports/REPOSITORY_REVIEW_20261030.md)
- [CHARTER.md](CHARTER.md)
- [Requirements traceability](REQUIREMENTS_TRACEABILITY.md)
- [Product and cost plan](reports/PRODUCT_AND_COST_PLAN.md)
- [Shortcut-analysis plan](evaluation/SHORTCUT_ANALYSIS_PLAN.md)
- [Agent design](agents/AGENT_DESIGN.md)
- [Agent evaluation plan](evaluation/AGENT_EVAL_PLAN.md)
- [Wrist Review app overview](docs/app/Wrist_Review_App.md)
- [Historical Phase 1 summary](reports/PHASE1_SUMMARY.md)
