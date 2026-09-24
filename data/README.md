# MURA data access

This repository never stores raw MURA files in Git. Obtain the Stanford MURA dataset through its official non-commercial research access route and honor its applicable data-use terms.

Set a local path before running the pipeline:

```powershell
$env:MURA_DATA_ROOT = "G:\datasets\MURA-v1.1"
```

The path must contain the official MURA structure, including `train/`, `valid/`, `train_image_paths.csv`, `valid_image_paths.csv`, `train_labeled_studies.csv`, and `valid_labeled_studies.csv`. Run `python scripts/audit_data.py --data-root $env:MURA_DATA_ROOT` first. The script creates local manifests under `data/manifests/`; those files are reproducibility artifacts but are excluded from Git because they reveal the local dataset inventory.

The project uses only `XR_WRIST` and predicts MURA's study-level NORMAL/ABNORMAL label. It is not a fracture detector and does not provide autonomous clinical diagnosis.

