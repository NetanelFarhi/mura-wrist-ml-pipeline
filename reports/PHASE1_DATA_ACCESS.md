# Phase 1 MURA data access

> **Current governance note:** “sealed final test” below describes the Phase-1 protocol at the time. Official MURA `valid` was subsequently observed once for that frozen reference evaluation and is **not** untouched for later model, threshold, calibration, architecture, or agent selection. See [TEST_SET_GOVERNANCE.md](TEST_SET_GOVERNANCE.md).

## Executed state

The authorized official Stanford MURA release was made available locally for this Phase 1 run. The project indexed only `XR_WRIST`; it did not substitute FracAtlas or another dataset. The executed audit produced local-only wrist and split manifests (excluded from Git) plus the safe aggregate [wrist_data_audit.json](wrist_data_audit.json).

The measured XR_WRIST scope is 3,474 patients, 3,697 studies, and 10,411 images. The official MURA `valid` source was retained as the sealed 237-study final test set. Raw MURA files remain local and are not included in the repository.

## Reproduction and data-use boundary

To repeat the pipeline in an authorized environment, point `MURA_DATA_ROOT` at a locally extracted official `MURA-v1.1` directory containing `train/` and `valid/`, then run:

```powershell
$env:MURA_DATA_ROOT = "<authorized-local-path-to-MURA-v1.1>"
cd <local-project-directory>
python scripts\audit_data.py --data-root $env:MURA_DATA_ROOT
```

The audit reads image files to verify them and writes only local project manifests/reports. Do not upload the dataset or route it through an unauthorized third party. Do not replace MURA with another dataset unless the project owner explicitly changes the research question and documents the consequence.
