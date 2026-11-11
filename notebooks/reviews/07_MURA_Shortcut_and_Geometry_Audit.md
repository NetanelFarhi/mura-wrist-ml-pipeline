# Review: `07_MURA_Shortcut_and_Geometry_Audit.ipynb`

[Open notebook source](../07_MURA_Shortcut_and_Geometry_Audit.ipynb)

**Classification:** Diagnostic 07
**Purpose:** Runs a fixed-model geometry/shortcut robustness audit on internal data only.

## Execution and prerequisites

Consumes authorized MURA train images, clean train/validation manifests, canonical Model B checkpoint/predictions/policy from `DATA/MODEL_B_FINE_TUNING/`, and `scripts/run_shortcut_audit.py`; writes audit report and summary tables under its configured output directory.

Static source review; 5 code cells; 0 unexecuted according to saved notebook metadata; 0 serialized error outputs. This is not a fresh run.

## Findings and maintenance notes

No fine-tuning or threshold selection. No serialized error output in saved state. Requires the full existing artifact/data layout and working repository root.

## Explicit file names found in notebook source/output

This list includes source references and generated artifact names; external runtime paths may resolve outside Git.
- `scripts/run_shortcut_audit.py`
- `wrist_internal_train_clean.csv`
- `wrist_internal_val_clean.csv`
- `best_resnet18_study_level.pt`
- `validation_predictions.csv`
- `frozen_primary_operating_policy.json`
- `audit_report.json`
- `robustness_summary.csv`
- `padding_subgroups.csv`
