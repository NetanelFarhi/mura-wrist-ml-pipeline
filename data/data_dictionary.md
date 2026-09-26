# Phase 1 data dictionary

> **Current governance note:** `test` below denotes the historical frozen Phase-1 reference split. Official MURA `valid` has been observed once and is not untouched for later model, threshold, calibration, architecture, or agent selection; see [`../reports/TEST_SET_GOVERNANCE.md`](../reports/TEST_SET_GOVERNANCE.md).

## `wrist_manifest.csv`

| Column | Type | Meaning | Source / transformation |
|---|---|---|---|
| `patient_id` | string | MURA patient directory identifier | Parsed from the official relative image path. |
| `study_id` | string | Stable patient-plus-study identifier | Parsed from the official relative image path; unique within the project. |
| `image_path` | string | Portable path to one image view | Stored relative to the configured MURA root when possible; resolved at runtime. |
| `label` | integer | `0` for NORMAL and `1` for ABNORMAL | Parsed from the study directory / official labeled-study CSV and checked for agreement. |
| `source_split` | string | `train` or `valid` | Official MURA origin before project splitting. |
| `num_views_in_study` | integer | Number of indexed image files for the study | Grouped count after the XR_WRIST index is built. |

## `split_manifest.csv`

| Column | Type | Meaning | Source / transformation |
|---|---|---|---|
| `patient_id` | string | MURA patient directory identifier | Copied from the wrist manifest. |
| `study_id` | string | Stable study identifier | Copied from the wrist manifest. |
| `final_split` | string | `train`, `val`, or `test` | Deterministic patient-level allocation. Official valid is sealed `test` by default. |

## Derived artifacts

Embedding caches contain a 512-dimensional frozen ResNet-18 feature vector per image plus its study metadata. Study embeddings are the arithmetic mean of image embeddings in one study. Evaluation outputs contain only study-level probabilities and labels.
