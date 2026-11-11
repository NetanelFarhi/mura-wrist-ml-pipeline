# Review: `01_MURA_Data_Preparation_and_EDA.ipynb`

[Open notebook source](../01_MURA_Data_Preparation_and_EDA.ipynb)

**Classification:** Canonical 01/06
**Purpose:** Creates the current XR_WRIST internal patient-level split from official MURA train.

## Execution and prerequisites

Requires authorized MURA train CSVs under `DATA/MURA-v1.1` or `MURA_DATA_ROOT`; writes train/validation manifests and a split summary under `WRIST_SPLITS`. Rerunning regenerates the split, so reuse the current persisted manifests when continuing existing results.

Static source review; 6 code cells; 0 unexecuted according to saved notebook metadata; 0 serialized error outputs. This is not a fresh run.

## Findings and maintenance notes

No blocking source issue found. Six code cells have stored execution counts and no serialized error output; this is repository state, not a new execution. The notebook mounts Drive and resolves its own paths; save generated files because notebook variables do not persist between notebooks.

## Explicit file names found in notebook source/output

This list includes source references and generated artifact names; external runtime paths may resolve outside Git.
- `train_labeled_studies.csv`
- `train_image_paths.csv`
- `wrist_internal_train.csv`
- `wrist_internal_val.csv`
- `wrist_internal_studies.csv`
- `data_preparation_summary.json`
