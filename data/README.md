# MURA data governance and access

This repository never stores raw MURA files in Git, including in a private repository. Obtain Stanford MURA through its official research/non-commercial access route and honor applicable data-use terms. Never commit credentials, access tokens, or protected data.

Set a local path before running the pipeline:

```powershell
$env:MURA_DATA_ROOT = "G:\datasets\MURA-v1.1"
```

The path must contain the official MURA structure, including `train/`, `valid/`, `train_image_paths.csv`, `valid_image_paths.csv`, `train_labeled_studies.csv`, and `valid_labeled_studies.csv`. Run `python scripts/audit_data.py --data-root $env:MURA_DATA_ROOT` first. The script creates full local manifests under `data/manifests/`; those files are reproducibility artifacts but are excluded from Git because they reveal the local dataset inventory.

Expected local layout is `MURA_DATA_ROOT/train/` and `MURA_DATA_ROOT/valid/` with the official CSVs. The repository may contain only access scripts, a data dictionary, schemas, and intentionally reviewed aggregate summaries; it never contains full local manifests or raw images. The project uses only `XR_WRIST` and predicts MURA's study-level NORMAL/ABNORMAL label. It is not a fracture detector and does not provide autonomous clinical diagnosis.

## Checkpoint 1 Colab workflow

The current Colab notebooks (01–07) do not use `MURA_DATA_ROOT`; they read the MURA ZIP and the derived files from the team's shared Google Drive folder `DATA/`. Its layout and files are described in [data_dictionary.md](data_dictionary.md) and in the root [README](../README.md).
