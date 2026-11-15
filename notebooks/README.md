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

## Diagnostics, runtime viewer, and preserved policy notebooks

- [07 — Shortcut/geometry audit](07_MURA_Shortcut_and_Geometry_Audit.ipynb): diagnostic only; reads the frozen Model B artifacts and does not train or select a threshold.
- [08 — Runtime study viewer](08_MURA_Runtime_Study_Viewer.ipynb): optional local preview of authorized internal-train studies. Keep preview disabled and clear outputs before commit.
- [09 — Deterministic Agent Policy](legacy/MURA_Deterministic_Agent_Policy.ipynb): preserved policy demonstration. It uses synthetic inputs and does not load a checkpoint or run TTA/Grad-CAM. The active deterministic logic is maintained under `tools/` and `src/`.
- [09 — Deterministic safety tools](legacy/09_MURA_Deterministic_Safety_Tools.ipynb): preserved supporting smoke-test notebook, retained unchanged.
- [Saved results viewer](06_results_viewer.ipynb): read-only presentation, separate from the execution sequence.
- [09 - MURA Agentic AI](09_MURA_Agentic_AI-final.ipynb): Completed and successfully executed end-to-end. Integrates frozen Model B, deterministic policy, authorized TTA and Grad-CAM tools, CrewAI-based research explanations, and safety controls. Three structured research artifacts were validated. Final notebook uploaded to GitHub; research use only, with mandatory human review.
- [09A — Independent Model B Verification](09A_MURA_Ground_True_Independent_Model_B_Validation.ipynb): Independently verifies frozen Model B predictions across three XR_WRIST studies (False Positive, True Positive, and True Negative). Reproduces all eight image-level predictions, verifies original MURA image identity, analyzes image-level disagreement, and documents Grad-CAM visualization and previously recorded agentic tool execution. Successfully rerun in a fresh Colab runtime without retraining or executing CrewAI; research use only.
- [10 — Official Valid Final Evaluation](10_MURA_Official_Valid_Final_Evaluation.ipynb): single pre-registered evaluation of frozen Model B and the deterministic decision layer on official MURA `valid` (XR_WRIST, 237 studies). Run once on 14.11.2026 after [PREREGISTRATION.md](../PREREGISTRATION.md) was committed; nothing is retrained or tuned. Reports sensitivity, specificity, AUROC and routing counts with patient-cluster bootstrap confidence intervals; results are in [reports/official_valid_final/report.md](../reports/official_valid_final/report.md). Official sensitivity is 89.7% (95% CI 82.8–95.3%), below the 95% target in the point estimate. This is repeat evaluation of a previously observed set. The LLM explanation layer was not run.

## Notebook 09: Agentic AI research workflow

[09_MURA_Agentic_AI-final.ipynb](09_MURA_Agentic_AI-final.ipynb) integrates the canonical frozen Model B, deterministic policy, optional TTA/Grad-CAM tools and CrewAI/Gemini research explanations. Its canonical policy uses mean view probabilities and threshold `0.263567715883255`; use the [frozen policy](../configs/agentic_ai/frozen_primary_operating_policy.json) and [bundle specification](../configs/model_b_artifact_bundle.json) for the authoritative settings.

- **Inputs:** authorized wrist studies and the verified external Model B bundle. No retraining or threshold selection is part of this workflow.
- **Recorded outputs:** three schema-validated [research artifacts](../reports/agentic_ai/research_artifacts/) and three [PDF reports](../reports/agentic_ai/research_reports/). RANDOM_001 and RANDOM_002 record successful TTA/Grad-CAM execution; NO_TOOLS_001 records neither optional tool being executed.
- **Evidence boundary:** selected-case technical demonstrations, not population-level clinical validation. The two tool-enabled PDF reports include token-based API cost estimates; the no-tools report has no saved token/cost metrics.
- **Execution boundary:** the notebook contains development/debugging cells and historical narrative. Review those separately from the validated artifact workflow; this documentation update does not certify a fresh sequential run or resolve all retained historical text. Use the canonical policy above rather than an inconsistent threshold in an older narrative block.
- **Safety:** explanations cannot change frozen predictions or deterministic routes. TTA measures stability, and Grad-CAM shows attribution; neither establishes diagnostic correctness. Human review is mandatory.

[Notebook 09A](09A_MURA_Ground_True_Independent_Model_B_Validation.ipynb) verifies three selected cases and reproduces their saved predictions. Here “independent” refers to verification of those cases, not an untouched independent evaluation cohort.

## Notebook 10: official-validation final evaluation

[10_MURA_Official_Valid_Final_Evaluation.ipynb](10_MURA_Official_Valid_Final_Evaluation.ipynb) evaluates the frozen Model B and deterministic policy v1.1.0. It does **not** execute Notebook 09, CrewAI, or the LLM explanation layer.

The notebook verifies the repository commit, artifact fingerprints and threshold, saves inference outputs before computing metrics, applies QC/routing, and writes `summary.json` and `report.md`. `MODE="internal"` supports the reproduction check; `MODE="official"` is the completed single official run and refuses a repeat. Internal and official checks use the same CPU device type. Do not rerun official evaluation as a setup step.

| Recorded result (14 November 2026) | Value |
| --- | --- |
| Official cohort | 237 studies, 207 patients; 97 ABNORMAL, 140 NORMAL |
| Sensitivity | 89.7% (95% CI 82.8–95.3%) |
| Specificity / AUROC | 64.3% / 0.903 |
| False negatives / false positives | 10 / 50 |
| Priority / routine routes | 159 / 78 |
| ABNORMAL studies remaining in routine | 7 |
| Execution checks | 38/38 PASS |

The sensitivity point estimate is below the 95% target, while its interval includes 95%. Confidence intervals use 1,000 patient-cluster bootstrap resamples. The official population had prior Phase 1 and selected-case exposure; this is not an untouched test. Model classification and priority routing are separate outcomes.

The first attempt stopped during manifest parsing before model image reads. The documented four-line parsing fix preceded the completed run; the model, threshold, rule and metrics were unchanged. See the [pre-registration](../PREREGISTRATION.md), [final report](../reports/official_valid_final/report.md) and [summary](../reports/official_valid_final/summary.json). Per-study files remain outside Git.

## Preserved contributor and experimental notebooks

- [Legacy notebooks](legacy/README.md) are kept for provenance. Hagit's Agentic AI draft is archived there unchanged and is not part of the active runtime.
- [All-anatomy extension](experimental_all_anatomies/README.md) is experimental only and is not the XR_WRIST MVP.

## Configuration and required files

The canonical Model B bundle is stored outside Git at `DATA/MODEL_B_FINE_TUNING/`; see [bundle metadata](../configs/model_b_artifact_bundle.json) and [artifact notes](../models/README.md). `configs/data.yaml`, `configs/frozen_logistic.yaml`, and `configs/resnet18_finetune.yaml` are supporting script/config files; the numbered notebooks also set several paths and run parameters directly in their cells. The bundle JSON is the authoritative file list and SHA-256 for the canonical Model B checkpoint.

## Per-notebook review notes

See the [per-notebook review index](reviews/README.md) for one separate audit note per notebook. These are **static reviews**: notebooks were not re-executed in this task. Existing stored cell outputs/counts are described as repository state, not as a new run. No fine-tuning was run.
