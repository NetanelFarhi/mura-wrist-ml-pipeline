# Review: `04_MURA_Baseline_Evaluation.ipynb`

[Open notebook source](../04_MURA_Baseline_Evaluation.ipynb)

**Classification:** Canonical 04/06  
**Purpose:** Evaluates saved Model A predictions on internal validation.

## Execution and prerequisites

Consumes `BASELINE_MODEL_A/validation_predictions.csv`, ranking metrics, selection and run metadata from Notebook 03; writes threshold tables, bootstrap intervals, plots, error summaries and evaluation README under `internal_validation_evaluation`.

Static source review; 14 code cells; 0 unexecuted according to saved notebook metadata; 0 serialized error outputs. This is not a fresh run.

## Findings and maintenance notes

No serialized error output in saved state. Depends on Notebook 03 outputs and does not load official MURA valid for a new decision. It mounts Drive and resolves its own model directory.

## Explicit file names found in notebook source/output

This list includes source references and generated artifact names; external runtime paths may resolve outside Git.
- `validation_predictions.csv`
- `validation_ranking_metrics.json`
- `model_selection.json`
- `run_metadata.json`
- `logistic_regression_candidates.csv`
- `internal_validation_roc_pr.png`
- `sensitivity_specificity_workload_tradeoff.png`
- `internal_validation_confusion_matrices.png`
- `operating_points.csv`
- `threshold_table.csv`
- `selected_primary_operating_point.json`
- `bootstrap_confidence_intervals.csv`
- `primary_policy_predictions.csv`
- `primary_policy_false_negatives.csv`
- `primary_policy_false_positives.csv`
- `primary_policy_by_view_count.csv`
- `evaluation_protocol.json`
- `evaluation_summary.md`
