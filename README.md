# MURA XR_WRIST Research Project

Course project for study-level **NORMAL / ABNORMAL** classification of wrist radiograph studies from Stanford MURA v1.1.

The project is a research prototype for decision support before specialist review. It predicts the MURA study label, which is an abnormality label — **not** a fracture diagnosis, fracture location, or autonomous clinical decision.

## Current checkpoint status

The current Checkpoint 1 path is working end to end in Google Colab:

```text
Patient-safe internal manifests
        ↓
02 preprocessing and image-quality checks
        ↓
03 frozen ResNet-18 embeddings + Logistic Regression
        ↓
04 internal-validation evaluation and error tables
```

The baseline is deliberately simple: each X-ray is converted to a 512-dimensional embedding by a frozen ImageNet ResNet-18, embeddings are mean-pooled across the views of a study, and Logistic Regression produces one study-level probability.

| Current run | Train | Internal validation |
|---|---:|---:|
| Images | 7,810 | 1,942 |
| Studies | 2,769 | 691 |
| Patients | 2,614 | 653 |
| NORMAL studies | 1,712 | 422 |
| ABNORMAL studies | 1,057 | 269 |

## Current baseline result — development validation only

Notebook 03 selected Logistic Regression `C = 0.1` from the pre-defined candidates `0.1, 1, 10` using the internal validation split.

| Metric | Result |
|---|---:|
| AUROC | 0.880442 |
| AUPRC | 0.851513 |

For the current pre-specified high-sensitivity policy, Notebook 04 selects the **highest validation threshold** that reaches sensitivity ≥ 90%.

| Operating-point metric | Result |
|---|---:|
| Threshold | 0.127701 |
| Sensitivity | 0.903346 |
| Specificity | 0.530806 |
| Precision | 0.551020 |
| F1 | 0.684507 |
| False negatives | 26 |
| False positives | 198 |
| Studies flagged for review | 441 / 691 (63.8%) |

These are real outputs saved in the executed notebooks, but they are **development results**: the same internal-validation population was used for model selection and threshold selection. They must not be presented as a final untouched-test result or as evidence of clinical readiness.

## Notebook workflow

| Notebook | Status | Purpose |
|---|---|---|
| `00_data_access_audit.ipynb` | Legacy scaffold | Local-folder data audit; not the current Colab ZIP workflow. |
| `01_eda.ipynb` | Legacy scaffold | Earlier local EDA template; a current Colab data-preparation/EDA notebook is still needed. |
| [02_preprocessing.ipynb](notebooks/02_preprocessing.ipynb) | Executed | Cleans the internal manifests, checks decode validity, preserves study grouping, and defines deterministic evaluation preprocessing. |
| [03_MURA_Baseline_ResNet18_LogReg.ipynb](notebooks/03_MURA_Baseline_ResNet18_LogReg.ipynb) | Executed | Extracts frozen ResNet-18 embeddings, pools them by study, fits Logistic Regression, and saves internal-validation predictions. |
| [04_MURA_Model_Evaluation.ipynb](notebooks/04_MURA_Model_Evaluation.ipynb) | Executed | Selects the validation-only operating point; creates ROC/PR curves, confusion matrices, bootstrap intervals, and false-positive/false-negative tables. |

Notebooks 03 and 04 use only the clean manifests produced by Notebook 02. They do not read the official MURA `valid` split.

## Data and Git policy

Raw MURA images, ZIP archives, per-study predictions, embeddings, checkpoints, credentials, and local Drive paths must never be committed. The repository contains code, notebook logic, reviewed notebook outputs, documentation, schemas, and aggregate results only.

For local data access and expected layout, see:

- [data/README.md](data/README.md)
- [data/data_dictionary.md](data/data_dictionary.md)
- [.gitignore](.gitignore)

The current Colab notebooks expect this local Google Drive structure:

```text
MyDrive/DATA/
├── MURA-v1.1_files.zip
├── WRIST_SPLITS/
│   ├── wrist_internal_train.csv
│   └── wrist_internal_val.csv
├── PREPROCESSED_WRIST/       # created by Notebook 02
└── BASELINE_MODEL_A/          # created by Notebooks 03 and 04
```

## Evaluation governance

The project uses patient-safe internal train/validation manifests. A patient and a study may not appear in both internal splits, and all views from one study stay together.

The official MURA `valid` split was used once for the historical Phase 1 frozen-reference evaluation. Its results have therefore already been observed and it cannot honestly be described as untouched for future architecture, threshold, calibration, or agent selection. The current Colab baseline run intentionally avoids it.

See [reports/TEST_SET_GOVERNANCE.md](reports/TEST_SET_GOVERNANCE.md) for the full historical record and constraints.

## Scope and next work

Implemented now:

- deterministic preprocessing and integrity checks;
- a study-level frozen ResNet-18 + Logistic Regression baseline;
- validation-only threshold analysis, bootstrap confidence intervals, and observable error tables.

Not implemented yet:

- current Colab data-preparation/EDA notebook that creates and documents the internal split;
- fine-tuned ResNet-18 comparison using the same study-level protocol;
- calibration, learned multi-view aggregation, or an external validation study;
- Grad-CAM diagnostic analysis;
- Agentic AI, RAG, Streamlit, or clinical deployment.

The Agentic AI layer is planned as a bounded, separately evaluated component. It is not part of the current baseline result. Its design and evaluation plan are documented in [agents/AGENT_DESIGN.md](agents/AGENT_DESIGN.md) and [evaluation/AGENT_EVAL_PLAN.md](evaluation/AGENT_EVAL_PLAN.md).

## Important limitations

- MURA `ABNORMAL` does not mean acute fracture.
- The 90% sensitivity operating point substantially increases review workload; it is a research tradeoff, not a clinical policy.
- Probability values are model scores, not automatically calibrated confidence.
- No model in this repository is ready for clinical use.

## Project documentation

- [CHARTER.md](CHARTER.md)
- [Requirements traceability](REQUIREMENTS_TRACEABILITY.md)
- [Product and cost plan](reports/PRODUCT_AND_COST_PLAN.md)
- [Shortcut-analysis plan](evaluation/SHORTCUT_ANALYSIS_PLAN.md)
- [Historical Phase 1 summary](reports/PHASE1_SUMMARY.md)
