# Review: `03_MURA_Baseline_ResNet18_LogReg.ipynb`

[Open notebook source](../03_MURA_Baseline_ResNet18_LogReg.ipynb)

**Classification:** Canonical 03/06
**Purpose:** Fits the frozen-image-embedding Model A baseline; this is not Model B fine-tuning.

## Execution and prerequisites

Consumes Notebook 02 clean manifests/spec and authorized MURA image archive; writes study embeddings/metadata, the Logistic Regression model, validation predictions, metrics, selection and run metadata under `BASELINE_MODEL_A`. Requires PyTorch/torchvision, scikit-learn, joblib and model weights.

Static source review; 16 code cells; 0 unexecuted according to saved notebook metadata; 0 serialized error outputs. This is not a fresh run.

## Findings and maintenance notes

This is an actual baseline fit and may download ImageNet weights. No serialized error output in saved state. Keep its output directory consistent with Notebook 04/06; current config YAML is not a shared live parameter store.

## Explicit file names found in notebook source/output

This list includes source references and generated artifact names; external runtime paths may resolve outside Git.
- `MURA-v1.1_files.zip`
- `wrist_internal_train_clean.csv`
- `wrist_internal_val_clean.csv`
- `preprocessing_spec.json`
- `self.zip`
- `zipfile.Zip`
- `_study_metadata.csv`
- `validation_predictions.csv`
- `validation_ranking_metrics.json`
- `model_selection.json`
- `logistic_regression_candidates.csv`
- `run_metadata.json`
