# Notebook guide

This guide identifies the current workflow, supporting viewers, experiments, and preserved contributor work. Notebook source was not changed during this organization.

## Canonical wrist workflow: 01–06

1. [01 — Data preparation and EDA](01_MURA_Data_Preparation_and_EDA.ipynb): builds the seed-42 patient-grouped split from MURA train.
2. [02 — Preprocessing](02_MURA_Preprocessing.ipynb): creates clean manifests and preprocessing metadata.
3. [03 — Model A baseline](03_MURA_Baseline_ResNet18_LogReg.ipynb): fits the frozen-image-embedding Logistic Regression baseline.
4. [04 — Model A evaluation](04_MURA_Baseline_Evaluation.ipynb): evaluates saved internal-validation predictions.
5. [05 — Model B](05_MURA_FineTuned_ResNet18.ipynb): default mode is `verify_existing`. Keep it there; do not select `train_new_experiment` for this task. It verifies the existing bundle and does not retrain in the default mode.
6. [06 — Model B comparison](06_MURA_ModelB_Evaluation_and_Comparison.ipynb): verifies the bundle and compares Model A/B on the current internal validation split.

Intermediate artifacts must be persisted in the configured Drive/data folders; Jupyter variables do not carry from one notebook to another. Paths can be set with each notebook's documented `MURA_*` environment variables. Notebook 01 regenerates the split if rerun, so retain the current manifests when continuity with existing results matters.

## Diagnostics and active deterministic runtime

- [07 — Shortcut/geometry audit](07_MURA_Shortcut_and_Geometry_Audit.ipynb): diagnostic only; reads the frozen Model B artifacts and does not train or select a threshold.
- [08 — Runtime study viewer](08_MURA_Runtime_Study_Viewer.ipynb): optional local preview of authorized internal-train studies. Keep preview disabled and clear outputs before commit.
- [09 — Deterministic Agent Policy](09_MURA_Deterministic_Agent_Policy.ipynb): new active policy demonstration. Uses synthetic inputs, the current deterministic safety module and `tools/agent_policy.py` v1.1; it does not load a checkpoint or run TTA/Grad-CAM.
- [09 — Deterministic safety tools](09_MURA_Deterministic_Safety_Tools.ipynb): existing supporting smoke-test notebook, retained unchanged.
- [Saved results viewer](06_results_viewer.ipynb): read-only presentation, separate from the execution sequence.

## Preserved contributor and experimental notebooks

- [Legacy notebooks](legacy/README.md) are kept for provenance. Hagit's Agentic AI draft is archived there unchanged and is not part of the active runtime.
- [All-anatomy extension](experimental_all_anatomies/README.md) is experimental only and is not the XR_WRIST MVP.

## Configuration and required files

The canonical Model B bundle is stored outside Git at `DATA/MODEL_B_FINE_TUNING/`; see [bundle metadata](../configs/model_b_artifact_bundle.json) and [artifact notes](../models/README.md). `configs/data.yaml`, `configs/frozen_logistic.yaml`, and `configs/resnet18_finetune.yaml` are supporting script/config files; the numbered notebooks also set several paths and run parameters directly in their cells. The bundle JSON is the authoritative file list and SHA-256 for the canonical Model B checkpoint.

## Per-notebook review notes

See the [per-notebook review index](reviews/README.md) for one separate audit note per notebook. These are **static reviews**: notebooks were not re-executed in this task. Existing stored cell outputs/counts are described as repository state, not as a new run. No fine-tuning was run.
