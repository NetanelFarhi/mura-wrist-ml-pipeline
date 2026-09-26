# MURA XR_WRIST Research Project

MURA XR_WRIST is a reproducible research pipeline for study-level NORMAL versus ABNORMAL classification of Stanford MURA `XR_WRIST` radiographs. Its historical Phase-1 work establishes whether fine-tuning an ImageNet-pretrained ResNet-18 materially improves on a cheaper frozen-feature Logistic Regression baseline before any advanced modelling or agentic workflow is considered.

## Current status

**Historical Phase 1 complete; later development is validation-only.** The project indexed and audited the authorized MURA XR_WRIST data, created patient-safe splits, trained both planned models, froze validation-only selections, and ran one official-valid Phase-1 reference evaluation. The executed scope contains 3,474 patients, 3,697 studies, and 10,411 images. Because official MURA `valid` results have now been observed, it is not untouched for subsequent model selection; see [test-set governance](reports/TEST_SET_GOVERNANCE.md). Raw images, derived predictions, embedding caches, checkpoints, and credentials remain local; reviewed documentation and non-sensitive aggregate reports may be versioned.

## Problem, user, and boundary

The intended user is an Emergency Department clinician reviewing musculoskeletal radiographs before specialist review. This is a research decision-support baseline only. MURA's labels represent NORMAL/ABNORMAL studies, not fracture diagnoses. The project does not claim fracture diagnosis, validated localization, clinical outcome improvement, or autonomous medical diagnosis. A clinician remains the final decision maker.

The repository preserves Phase 1 and later development artifacts. It does **not** implement the Agentic AI layer, RAG, Streamlit, Grad-CAM, ensemble, or a clinical deployment. Phase 2A/2B are development experiments and are not untouched-test claims; see test-set governance and phase reports.

## Dataset and prediction unit

The pipeline uses only the official [Stanford MURA dataset](https://aimi.stanford.edu/datasets/mura-msk-xrays), restricted to `XR_WRIST`.

```text
Patient
  └── Study
        └── One or more X-ray image views
```

The final unit of prediction and evaluation is a **study**. Images from the same patient stay in one split and all views from a study stay together. The original protocol used official MURA validation once as frozen Phase-1 reference after patient-safe train/validation development; it is now observed and cannot be used for later selection.

## Models

| Model | Training strategy | Study aggregation | Purpose |
|---|---|---|---|
| A: Frozen embedding baseline | ImageNet ResNet-18 is frozen; a 512-dimensional image embedding is extracted; study embeddings are mean-pooled; Logistic Regression performs a small validation-only `C=[0.1, 1, 10]` search. | Mean image embedding | Cheapest reasonable comparison required by the course. |
| B: Fine-tuned ResNet-18 | ImageNet ResNet-18 is fine-tuned on wrist views with safe augmentation, early stopping, and optional class weighting only when imbalance warrants it. | Mean and max probabilities compared on validation only | Main Phase 1 deep-learning model. |

The Phase-1 reference evaluation did not use final-test data to select hyperparameters, preprocessing, aggregation, checkpoints, or threshold. Subsequent work must not use the already-observed official-valid results for selection.

## Evaluation and leakage prevention

Every result is computed at study level. Reported metrics are AUROC, AUPRC, sensitivity/recall, specificity, F1, false-negative rate, and a confusion matrix at threshold 0.5 plus a validation-selected operating threshold. AUROC bootstrap confidence intervals are supported when class counts permit.

Automated checks verify:

1. no patient or study overlap across final splits;
2. all views of each study remain together;
3. labels are internally consistent inside each study;
4. test data is excluded from fitting and threshold selection;
5. validation data is not mixed into training; and
6. preprocessing does not learn future/test statistics.

False-negative and false-positive reports focus on observable prediction and metadata patterns. They do not infer a medical cause for an error.

## Repository layout

```text
mura-x/
├── configs/                 # Fixed data and model settings
├── data/                    # Access notes, dictionary, generated local manifests
├── notebooks/               # Ordered executable workflow notebooks
├── src/
│   ├── data/                # Indexing, splitting, datasets, transforms
│   ├── features/            # Frozen embedding extraction
│   ├── models/              # Logistic baseline and ResNet-18 classifier
│   ├── training/            # Trainer, losses, metrics
│   └── evaluation/          # Leakage, bootstrap, metrics, errors
├── scripts/                 # CLI entry points
├── tests/                   # Synthetic-data integrity and smoke tests
├── models/                  # Local checkpoints, intentionally ignored
└── reports/                 # Figures, results, progress, and Phase 1 summary
```

Raw images, embedding caches, generated manifests, checkpoints, and secrets are excluded from Git.

## Reproduce Phase 1

Use a clean 64-bit Python 3.10 or 3.11 environment. The inspected machine has a GTX 1660 Ti but the available PyTorch runtime is CPU-only, so the scripts choose CPU safely unless a CUDA-enabled PyTorch build is installed.

These commands reproduce the historical Phase-1 protocol only. The lines referencing the official `valid`/`test` split must not be used to select or tune any later model, threshold, calibration, aggregation, or agent behavior; see [test-set governance](reports/TEST_SET_GOVERNANCE.md).

```powershell
cd C:\Users\netaf\Documents\Codex\2026-09-23\important-change-to-phase-1-phase\mura-x
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
```

For GPU use, install a PyTorch wheel compatible with the local NVIDIA driver from the official PyTorch installer, then verify `torch.cuda.is_available()` before running a full experiment.

After authorized MURA access:

```powershell
$env:MURA_DATA_ROOT = "G:\datasets\MURA-v1.1"
python scripts\audit_data.py --data-root $env:MURA_DATA_ROOT
# Model A: frozen ImageNet ResNet-18 features for all three fixed splits.
python scripts\extract_embeddings.py --manifest data\manifests\wrist_manifest.csv --split-manifest data\manifests\split_manifest.csv --split train --output data\processed\embeddings_train.npz --dataset-root $env:MURA_DATA_ROOT --allow-weight-download
python scripts\extract_embeddings.py --manifest data\manifests\wrist_manifest.csv --split-manifest data\manifests\split_manifest.csv --split val --output data\processed\embeddings_val.npz --dataset-root $env:MURA_DATA_ROOT --allow-weight-download
python scripts\extract_embeddings.py --manifest data\manifests\wrist_manifest.csv --split-manifest data\manifests\split_manifest.csv --split test --output data\processed\embeddings_test.npz --dataset-root $env:MURA_DATA_ROOT --allow-weight-download
python scripts\train_logistic_baseline.py --train-embeddings data\processed\embeddings_train.npz --val-embeddings data\processed\embeddings_val.npz --test-embeddings data\processed\embeddings_test.npz --write-test-predictions --output-dir models\frozen_logistic

# Model B: validation selects checkpoint, aggregation, and threshold.
python scripts\train_resnet18.py --manifest data\manifests\wrist_manifest.csv --split-manifest data\manifests\split_manifest.csv --output-dir models\resnet18 --dataset-root $env:MURA_DATA_ROOT --allow-weight-download --write-test-predictions

# The only final metric computation: a sealed study-level comparison and error analysis.
python scripts\evaluate_phase1.py --evaluation-config configs\evaluation_models.json --manifest data\manifests\wrist_manifest.csv --split-manifest data\manifests\split_manifest.csv --output-dir reports\metrics --error-output-dir reports\errors --baseline-report reports\baseline_results.md --error-report reports\error_analysis.md
```

The commands generate exact manifests, EDA figures, model artifacts, validation choices, sealed-test predictions, one study-level comparison table, and error-analysis reports. Final test metrics are computed only by the last command, after both models' validation decisions are frozen.

## Mandatory baseline table - historical Phase-1 reference

The table below is the once-opened 237-study official-valid Phase-1 reference evaluation. Both operating thresholds were selected and frozen on internal validation data before that evaluation. It is a fair same-population baseline comparison, not a new untouched advanced-model test.

| Model | AUROC | AUPRC | Sensitivity | Specificity | F1 | False-negative rate |
|---|---:|---:|---:|---:|---:|---:|
| Frozen ResNet-18 embeddings plus Logistic Regression | 0.876 | 0.860 | 0.825 | 0.807 | 0.784 | 0.175 (17/97) |
| Fine-tuned ResNet-18 | 0.897 | 0.894 | 0.732 | 0.950 | 0.811 | 0.268 (26/97) |

Fine-tuning improved overall discrimination and precision-recall ranking, while the frozen selected threshold traded higher specificity for more false negatives. This is a research baseline, not a clinical-deployment claim.

The future measurable operating target is sensitivity >=90%, reporting resulting specificity, FN rate, precision, F1, AUROC/AUPRC, and bootstrap CIs; it is not claimed as achieved on an untouched advanced-model test. The agent design, product plan, source checks, and traceability are documented in `agents/`, `evaluation/`, and `reports/`.

The concise entry point for preserved historical metrics is [reports/results.md](reports/results.md); the 4 October submission checklist is [reports/SUBMISSION_2026-10-04.md](reports/SUBMISSION_2026-10-04.md). Before committing, run `python scripts/check_repo_data_governance.py --staged` to prevent accidental addition of raw data or credentials.

## Course alignment and known limitations

The BIU DS23 guidelines require a clean, reproducible repository, a data dictionary, ordered notebooks, leakage documentation, error analysis, and a comparison with the simplest reasonable baseline. This repository implements the Phase 1 elements of those requirements. A later course phase must add and evaluate a justified, guarded agentic layer and complete assigned team roles; it is outside this scope.

Current limitations are the single 237-study sealed test set, no external validation, no clinically justified sensitivity target, and the absence of the later agentic workflow. Do not substitute FracAtlas automatically: it changes the target task and needs an explicit decision.

## Project roles

This workspace was requested for the lead Machine Learning Engineer and Data Scientist role. Named Data, Model, Agent, and Product ownership has not been supplied, so it must be recorded by the project team before final course submission.
