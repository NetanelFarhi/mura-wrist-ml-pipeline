# MURA data governance and access

This repository never stores raw MURA files in Git, including in a private repository. Obtain Stanford MURA through its official research/non-commercial access route and honor applicable data-use terms. Never commit credentials, access tokens, or protected data.

Set a local path before running the pipeline:

```powershell
$env:MURA_DATA_ROOT = "<local-path-to-MURA-v1.1>"
```

The path must contain the official MURA structure, including `train/`, `valid/`, `train_image_paths.csv`, `valid_image_paths.csv`, `train_labeled_studies.csv`, and `valid_labeled_studies.csv`. Run `python scripts/audit_data.py --data-root $env:MURA_DATA_ROOT` first. The script creates full local manifests under `data/manifests/`; those files are reproducibility artifacts but are excluded from Git because they reveal the local dataset inventory.

Expected local layout is `MURA_DATA_ROOT/train/` and `MURA_DATA_ROOT/valid/` with the official CSVs. The repository may contain only access scripts, a data dictionary, schemas, and intentionally reviewed aggregate summaries; it never contains full local manifests or raw images. The project uses only `XR_WRIST` and predicts MURA's study-level NORMAL/ABNORMAL label. It is not a fracture detector and does not provide autonomous clinical diagnosis.


## Evaluation populations

Internal training and validation are patient-grouped partitions of official MURA `train`. Official XR_WRIST `valid` contains 237 studies (140 NORMAL, 97 ABNORMAL) from 207 patients. It was evaluated in Phase 1, inspected in selected agent-workflow cases and evaluated again in the pre-registered final run on 14 November 2026. It must not be called an untouched locked test or used to select later models or thresholds. See [pre-registration](../PREREGISTRATION.md) and the [official-validation report](../reports/official_valid_final/report.md).
