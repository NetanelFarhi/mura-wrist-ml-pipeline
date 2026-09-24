# MURA X Phase 1

MURA X Phase 1 is a reproducible research pipeline for study-level NORMAL versus ABNORMAL classification of Stanford MURA `XR_WRIST` radiographs. It establishes whether fine-tuning an ImageNet-pretrained ResNet-18 materially improves on a cheaper frozen-feature Logistic Regression baseline before any advanced modelling or agentic workflow is considered.

## Current status

**PARTIAL.** The source documents have been reviewed; the independent pipeline is being implemented and smoke-tested. No authorized local MURA data was found, so no dataset counts, training runs, test metrics, threshold, or model comparison values are reported. See [reports/PHASE1_DATA_ACCESS.md](reports/PHASE1_DATA_ACCESS.md) for the one required data-access action and [reports/PHASE1_SUMMARY.md](reports/PHASE1_SUMMARY.md) for the current evidence.

## Problem, user, and boundary

The intended user is an Emergency Department clinician reviewing musculoskeletal radiographs before specialist review. This is a research decision-support baseline only. MURA's labels represent NORMAL/ABNORMAL studies, not fracture diagnoses. The project does not claim fracture diagnosis, validated localization, clinical outcome improvement, or autonomous medical diagnosis. A clinician remains the final decision maker.

This repository is deliberately **Phase 1 only**. It does not implement Agentic AI, RAG, Streamlit, Grad-CAM, uncertainty modelling, attention-based multi-view learning, ensembles, DenseNet, ConvNeXt, or EfficientNet experiments. The BIU course requires an agentic layer for the later final capstone; that work is explicitly deferred rather than misrepresented as complete.

## Dataset and prediction unit

The pipeline uses only the official [Stanford MURA dataset](https://aimi.stanford.edu/datasets/mura-msk-xrays), restricted to `XR_WRIST`.

```text
Patient
  └── Study
        └── One or more X-ray image views
```

The final unit of prediction and evaluation is a **study**. Images from the same patient stay in one split, all views from a study stay together, and the default protocol holds the official MURA validation split as a sealed final test set. A patient-safe training/validation partition is made from official training data only.

## Models

| Model | Training strategy | Study aggregation | Purpose |
|---|---|---|---|
| A: Frozen embedding baseline | ImageNet ResNet-18 is frozen; a 512-dimensional image embedding is extracted; study embeddings are mean-pooled; Logistic Regression performs a small validation-only `C=[0.1, 1, 10]` search. | Mean image embedding | Cheapest reasonable comparison required by the course. |
| B: Fine-tuned ResNet-18 | ImageNet ResNet-18 is fine-tuned on wrist views with safe augmentation, early stopping, and optional class weighting only when imbalance warrants it. | Mean and max probabilities compared on validation only | Main Phase 1 deep-learning model. |

The final test set is never used to select hyperparameters, preprocessing, aggregation, checkpoints, or an operating threshold.

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

## Current measured results

No MURA-dependent experiment has run. The comparison table is intentionally unpopulated:

| Model | AUROC | AUPRC | Recall | Specificity | F1 | FN Rate |
|---|---:|---:|---:|---:|---:|---:|
| Frozen ResNet-18 embeddings plus Logistic Regression | Not executed | Not executed | Not executed | Not executed | Not executed | Not executed |
| Fine-tuned ResNet-18 | Not executed | Not executed | Not executed | Not executed | Not executed | Not executed |

## Course alignment and known limitations

The BIU DS23 guidelines require a clean, reproducible repository, a data dictionary, ordered notebooks, leakage documentation, error analysis, and a comparison with the simplest reasonable baseline. This repository implements the Phase 1 elements of those requirements. A later course phase must add and evaluate a justified, guarded agentic layer and complete assigned team roles; it is outside this scope.

Current blockers are authorized MURA access and a CUDA-enabled PyTorch environment for practical full training. CPU smoke tests remain supported. Do not substitute FracAtlas automatically: it changes the target task and needs an explicit decision.

## Project roles

This workspace was requested for the lead Machine Learning Engineer and Data Scientist role. Named Data, Model, Agent, and Product ownership has not been supplied, so it must be recorded by the project team before final course submission.
