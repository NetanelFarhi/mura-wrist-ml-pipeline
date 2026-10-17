# Model artifacts

This directory is intentionally empty in source control. Training scripts write only local artifacts here:

- `frozen_logistic.joblib` and its metadata for Model A.
- `resnet18_best.pt` and its validation-selection metadata for Model B.

Model selection and threshold selection must use training and validation data only. Do not overwrite a final-test result with a new tuning run.

## Checkpoint 1 Colab artifacts

The Colab notebooks keep model artifacts on the shared Drive, never in Git: Model A in `DATA/BASELINE_MODEL_A/` (Notebook 03) and Model B checkpoints in `DATA/RUNS_ARTEM/05_MODEL_B_FINETUNE/run_*/best_model_b.pt` (Notebook 05). Notebook 06a records the SHA-256 of the frozen models in `operating_policy.json`; Notebooks 06b and 07 refuse to use a model whose hash differs.
