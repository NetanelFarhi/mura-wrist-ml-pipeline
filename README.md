# MURA XR_WRIST Research Project

Course project for study-level **NORMAL / ABNORMAL** classification of wrist
radiograph studies from Stanford MURA v1.1. This is a research prototype for
decision support before specialist review. It predicts the MURA study label;
it is **not** a fracture diagnosis, fracture-localization system, or autonomous
clinical decision tool.

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

Earlier notebooks are preserved under [notebooks/legacy](notebooks/legacy/)
for traceability. They are not the canonical workflow and must not be used to
make new selection decisions without protocol review.

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

- [CHARTER.md](CHARTER.md)
- [Requirements traceability](REQUIREMENTS_TRACEABILITY.md)
- [Product and cost plan](reports/PRODUCT_AND_COST_PLAN.md)
- [Shortcut-analysis plan](evaluation/SHORTCUT_ANALYSIS_PLAN.md)
- [Agent design](agents/AGENT_DESIGN.md)
- [Agent evaluation plan](evaluation/AGENT_EVAL_PLAN.md)
- [Historical Phase 1 summary](reports/PHASE1_SUMMARY.md)
