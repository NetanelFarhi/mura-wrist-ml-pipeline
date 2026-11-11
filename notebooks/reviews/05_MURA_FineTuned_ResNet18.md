# Review: `05_MURA_FineTuned_ResNet18.ipynb`

[Open notebook source](../05_MURA_FineTuned_ResNet18.ipynb)

**Classification:** Canonical 05/06; verification-only in this task  
**Purpose:** Validates the existing canonical Model B bundle; do not rerun fine-tuning.

## Execution and prerequisites

Consumes Notebook 02 clean manifests and the external `DATA/MODEL_B_FINE_TUNING/` bundle; reads `configs/model_b_artifact_bundle.json`; may reassemble existing `checkpoint_parts/`; verifies with `scripts/verify_model_artifact_bundle.py`.

Static source review; 7 code cells; 0 unexecuted according to saved notebook metadata; 0 serialized error outputs. This is not a fresh run.

## Findings and maintenance notes

The default `MURA_MODEL_B_RUN_MODE` is `verify_existing`. Keep this mode. If required artifacts are missing, it fails closed; the `train_new_experiment` branch is explicitly separate and was not run. Stored output is not a fresh verification.

## Explicit file names found in notebook source/output

This list includes source references and generated artifact names; external runtime paths may resolve outside Git.
- `scripts/train_resnet18.py`
- `model_b_artifact_bundle.json`
- `verify_model_artifact_bundle.py`
- `reassemble_model_b_checkpoint.py`
- `wrist_internal_train_clean.csv`
- `wrist_internal_val_clean.csv`
- `wrist_clean_manifest.csv`
- `wrist_clean_split_manifest.csv`
- `best_resnet18_study_level.pt`
- `checkpoint_parts.json`
- `training_history.json`
- `validation_metrics.json`
- `validation_predictions.csv`
- `validation_predictions_mean.csv`
- `validation_predictions_max.csv`
- `selection.json`
- `frozen_primary_operating_policy.json`
- `run reassemble_model_b_checkpoint.py`
- `run_config.json`
