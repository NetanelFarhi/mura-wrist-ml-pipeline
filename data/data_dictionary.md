# Data dictionary

The current Checkpoint 1 pipeline (Colab Notebooks 01–07) is described first. The Phase 1 local manifests are kept
below as a historical record.

## Checkpoint 1 Colab pipeline (Notebooks 01–07)

The Colab notebooks read one shared Google Drive folder, `DATA/`. Team members reach it through a `DATA` shortcut in
My Drive (`MyDrive/DATA`) or through the shared `Final Project DS23` shortcut
(`MyDrive/Final Project DS23/Agentic AI-Assisted Detection of Musculoskeletal Abnormalities in Emergency X-rays/DATA`).
Both are the same physical folder; Notebooks 01, 05, 06a, 06b and 07 accept either path, Notebooks 02–04 expect
`MyDrive/DATA`. None of the files below is committed to Git.

### `WRIST_SPLITS/` (written once by Notebook 01; later runs only verify)

| File | Unit | Columns | Notes |
|---|---|---|---|
| `mura_wrist_internal_split.csv` | study | `patient_id`, `study_uid`, `label`, `internal_split` | Source of truth for the patient-level 80/20 split (`StratifiedGroupKFold`, seed 42). |
| `wrist_internal_train.csv`, `wrist_internal_val.csv` | image | `file_id`, `file_name`, `size`, `added_at`, `md5_hash`, `file_extension`, `patient_id`, `study_id`, `label`, `study_uid` | `label` is text (`positive` / `negative`); `file_name` is the path inside the MURA ZIP. |
| `wrist_internal_val_clean.csv` | image | as above | Legacy (10.10.2026): validation without the 4 macOS `._` files. No longer written or checked by Notebook 01 and read by no notebook; the clean validation manifest used downstream is `PREPROCESSED_WRIST/wrist_internal_val_clean.csv` (Notebook 02). |
| `wrist_official_validation.csv` | image | `file_id`, `file_name`, `size`, `added_at`, `md5_hash` | Official MURA `valid`, XR_WRIST (659 images). Patient, study and label are encoded in `file_name`. |
| `wrist_official_validation_studies.csv` | study | `study_path`, `label` | Official study labels, numeric 0 / 1 (237 studies). |

### `PREPROCESSED_WRIST/` (Notebook 02)

`wrist_internal_train_clean.csv` and `wrist_internal_val_clean.csv` add `label_binary` (0 = NORMAL, 1 = ABNORMAL)
and keep only images that decode fully. `rejected_manifest_rows.csv` lists every removed row with its reason;
`study_view_audit.csv` and `preprocessing_summary.csv` count views, studies and patients per split.
`preprocessing_spec.json` records the deterministic transform (grayscale → pad to square → 224 × 224 → 3 channels →
ImageNet normalisation); `preprocessing_provenance.json` stores the SHA-256 of the input and output manifests.

### `BASELINE_MODEL_A/` (Notebooks 03 and 04)

Notebook 03: `frozen_resnet18_mean_embedding_logreg.joblib` (Model A), `validation_predictions.csv`, study embeddings
(`*_study_embeddings.npz`, `*_study_metadata.csv`), `validation_ranking_metrics.json`, `model_selection.json`,
`logistic_regression_candidates.csv`, `run_metadata.json`. Notebook 04 writes its evaluation (operating points,
threshold table, bootstrap intervals, error tables, figures) to `internal_validation_evaluation/`.

### Prediction files (Notebooks 03, 05, 06b)

| Column | Type | Meaning |
|---|---|---|
| `split` | string | `validation` (internal) — test files carry a `model` column instead |
| `patient_id` | string | MURA patient identifier |
| `study_id` | string | `patient<N>/study<N>_<label>` |
| `label` | integer | 0 = NORMAL, 1 = ABNORMAL (study level) |
| `num_views` | integer | Images in the study |
| `abnormal_probability` | float | Study-level model score used for thresholds (Model B: selected aggregation) |
| `prob_mean`, `prob_max` | float | Model B only: mean / max of view probabilities |

`validation_view_predictions.csv` (Model B) has one row per image with `view_probability`.

### `operating_policy.json` (Notebook 06a, frozen on internal validation)

Thresholds chosen on internal validation only: the highest threshold with sensitivity ≥ 95% (primary) and ≥ 90%
(reference, as Notebook 04), per model; the SHA-256 of each model artifact and validation-prediction file; the
Model B seed runs compared (`model_b_candidates`, `model_b_seed_summary`), the selected Model B run (highest
specificity at sensitivity ≥ 95%) and the pipeline model (Model B only if that specificity exceeds Model A's by
≥ 0.02, otherwise Model A). File paths are stored relative to the `DATA/` folder (`paths_relative_to`). The same
folder holds `validation_operating_points.csv` and `model_b_seed_runs.csv`. Notebooks 06b and 07 read models and
thresholds from this file only.

### Notebook 07 outputs (agent stub)

| File | Content |
|---|---|
| `case_reports.jsonl` | One `AgentReport` per internal-validation study, produced by `src/pipeline/agent_stub.py` (contract version 1.0): `case_id`, `model_flag` (`ABOVE_THRESHOLD` / `BELOW_THRESHOLD`), `agent_status` (`STUB`), `evidence` (the unchanged `ModelOutput`: `study_id`, `model_version`, `study_probability`, `threshold`, `view_probabilities` — `null` for Model A), `limitations`, `trace_id`, `contract_version`. |
| `live_end_to_end_check.csv` | Per checked study: `study_id`, `live_probability` (recomputed from the radiographs), `saved_probability`, `abs_difference`, `live_flag`, `saved_flag`. |
| `demo_case_report.json` | The full `AgentReport` of the demonstration study. |
| `pipeline_run_summary.json` | Policy run and SHA-256, pipeline model and version, threshold, number of reports and flag counts, live-check setting, number of studies and maximum difference, demonstration flag, path of `case_reports.jsonl` (relative to `DATA/`), `official_mura_valid_read = false`. |

### Run folders

Checkpoint-1 runs by Artem write only to new, time-stamped folders under `DATA/RUNS_ARTEM/`
(`05_MODEL_B_FINETUNE/`, `06A_OPERATING_POLICY/`, `06B_LOCKED_TEST/`, `07_AGENT_STUB_PIPELINE/`). Re-runs of
Notebooks 02–04 go to `02_PREPROCESSING/`, `03_BASELINE_MODEL_A/` and `04_MODEL_A_EVALUATION/` when their outputs
already exist in the shared folders. Existing team files are read-only.

## Historical Phase 1 local manifests

> **Governance note:** `test` below denotes the historical frozen Phase-1 reference split. Official MURA `valid` has been observed once and is not untouched for later model, threshold, calibration, architecture, or agent selection; see [`../reports/TEST_SET_GOVERNANCE.md`](../reports/TEST_SET_GOVERNANCE.md).

### `wrist_manifest.csv`

| Column | Type | Meaning | Source / transformation |
|---|---|---|---|
| `patient_id` | string | MURA patient directory identifier | Parsed from the official relative image path. |
| `study_id` | string | Stable patient-plus-study identifier | Parsed from the official relative image path; unique within the project. |
| `image_path` | string | Portable path to one image view | Stored relative to the configured MURA root when possible; resolved at runtime. |
| `label` | integer | `0` for NORMAL and `1` for ABNORMAL | Parsed from the study directory / official labeled-study CSV and checked for agreement. |
| `source_split` | string | `train` or `valid` | Official MURA origin before project splitting. |
| `num_views_in_study` | integer | Number of indexed image files for the study | Grouped count after the XR_WRIST index is built. |

### `split_manifest.csv`

| Column | Type | Meaning | Source / transformation |
|---|---|---|---|
| `patient_id` | string | MURA patient directory identifier | Copied from the wrist manifest. |
| `study_id` | string | Stable study identifier | Copied from the wrist manifest. |
| `final_split` | string | `train`, `val`, or `test` | Deterministic patient-level allocation. Official valid is sealed `test` by default. |

### Derived artifacts

Embedding caches contain a 512-dimensional frozen ResNet-18 feature vector per image plus its study metadata. Study embeddings are the arithmetic mean of image embeddings in one study. Evaluation outputs contain only study-level probabilities and labels.
