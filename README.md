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

## Checkpoint 1 results — development validation only

The current comparison uses the same internal, patient-level validation split:
691 studies from 653 patients (422 NORMAL, 269 ABNORMAL), drawn from official
MURA train. The official MURA `valid` split was not read for this run.

| Model | AUROC | AUPRC | Specificity at sensitivity ≥90% | Studies flagged for review |
|---|---:|---:|---:|---:|
| Frozen ResNet-18 embeddings + Logistic Regression | 0.880 | 0.852 | 0.531 | 63.8% |
| Fine-tuned ResNet-18, mean probability pooling | **0.913** | **0.878** | 0.706 | 53.1% |
| Fine-tuned ResNet-18, max probability pooling | 0.883 | 0.806 | 0.711 | 52.8% |

Mean pooling was selected by internal-validation AUROC. At the same observed
high-sensitivity operating point (90.3%; 26 false negatives), fine-tuned mean
pooling produced 124 false positives versus 198 for the frozen baseline. That
is a validation-only development finding, not a clinical claim or a final
untouched-test result.

See the [full Checkpoint 1 results page](docs/results/checkpoint1/README.md)
for ROC/precision–recall curves, training history, patient-cluster bootstrap
intervals, and the diagnostic shortcut/geometry audit.

## Data and evaluation governance

Raw MURA images, ZIP archives, per-study predictions, embeddings, checkpoints,
credentials, and local Drive paths must never be committed. See
[data/README.md](data/README.md), [data/data_dictionary.md](data/data_dictionary.md),
and [.gitignore](.gitignore) for the expected local data layout and safeguards.

All views from a study remain together, and a patient may not appear in both
internal train and validation. The official MURA `valid` split was used once
for the historical Phase 1 frozen-reference evaluation. Because that result has
already been observed, it is not an untouched test set for future architecture,
threshold, calibration, or agent decisions. The current Checkpoint 1 workflow
does not read it. See [reports/TEST_SET_GOVERNANCE.md](reports/TEST_SET_GOVERNANCE.md).

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
