# Review: `02_MURA_Preprocessing.ipynb`

[Open notebook source](../02_MURA_Preprocessing.ipynb)

**Classification:** Canonical 02/06
**Purpose:** Cleans manifests, audits decoding, and records deterministic preprocessing.

## Execution and prerequisites

Consumes `wrist_internal_train.csv` and `wrist_internal_val.csv` from Notebook 01 plus authorized `MURA-v1.1_files.zip`; writes clean manifests, audit tables, preprocessing spec and provenance under `PREPROCESSED_WRIST`. Requires pandas, NumPy, Pillow, PyTorch and torchvision.

Static source review; 10 code cells; 0 unexecuted according to saved notebook metadata; 0 serialized error outputs. This is not a fresh run.

## Findings and maintenance notes

No serialized error output in the saved notebook state. Must follow Notebook 01 and use the same split folder. It opens Drive and resolves paths within each notebook; keep output artifacts between sessions.

## Explicit file names found in notebook source/output

This list includes source references and generated artifact names; external runtime paths may resolve outside Git.
- `platform.py`
- `MURA-v1.1_files.zip`
- `wrist_internal_train.csv`
- `wrist_internal_val.csv`
- `wrist_internal_train_clean.csv`
- `wrist_internal_val_clean.csv`
- `rejected_manifest_rows.csv`
- `study_view_audit.csv`
- `preprocessing_summary.csv`
- `preprocessing_spec.json`
- `preprocessing_provenance.json`
