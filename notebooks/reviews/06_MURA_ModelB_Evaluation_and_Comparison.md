# Review: `06_MURA_ModelB_Evaluation_and_Comparison.ipynb`

[Open notebook source](../06_MURA_ModelB_Evaluation_and_Comparison.ipynb)

**Classification:** Canonical 06/06
**Purpose:** Compares Model A and Model B on the same current internal-validation cohort.

## Execution and prerequisites

Consumes Model A predictions from `BASELINE_MODEL_A`, canonical Model B predictions and policy from `DATA/MODEL_B_FINE_TUNING`, and bundle metadata/verifier; writes comparison tables, curves and bootstrap outputs under `MODEL_A_MODEL_B_COMPARISON`.

Static source review; 7 code cells; 0 unexecuted according to saved notebook metadata; 0 serialized error outputs. This is not a fresh run.

## Findings and maintenance notes

No serialized error output in saved state. Verifies Model B before comparing. Requires matching split manifests/prediction rows. Do not substitute historical `models/resnet18` artifacts.

## Explicit file names found in notebook source/output

This list includes source references and generated artifact names; external runtime paths may resolve outside Git.
- `verify_model_artifact_bundle.py`
- `model_b_artifact_bundle.json`
- `validation_predictions.csv`
- `_threshold_table.csv`
- `internal_validation_model_comparison.csv`
- `internal_validation_bootstrap_intervals.csv`
- `frozen_primary_operating_policy.json`
- `governance.json`
