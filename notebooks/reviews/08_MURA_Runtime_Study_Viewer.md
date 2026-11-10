# Review: `08_MURA_Runtime_Study_Viewer.ipynb`

[Open notebook source](../08_MURA_Runtime_Study_Viewer.ipynb)

**Classification:** Local utility 08  
**Purpose:** Displays views from one authorized internal-train study for local QC only.

## Execution and prerequisites

Consumes `DATA/MURA-v1.1` (or `MURA_DATA_ROOT`) and `WRIST_SPLITS/wrist_internal_train.csv`; requires pandas, Pillow and matplotlib.

Static source review; 2 code cells; 0 unexecuted according to saved notebook metadata; 0 serialized error outputs. This is not a fresh run.

## Findings and maintenance notes

Keep `SHOW_RUNTIME_PREVIEW=False`; clear notebook outputs before commit so radiograph pixels are not stored. This is not a model or evaluation step.

## Explicit file names found in notebook source/output

This list includes source references and generated artifact names; external runtime paths may resolve outside Git.
- `wrist_internal_train.csv`
