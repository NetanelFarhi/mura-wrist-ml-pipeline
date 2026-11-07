"""Paired Model B minus Model A comparison on the internal-validation cohort.

The operating threshold (sensitivity >= 95%) is re-derived inside every
patient-cluster bootstrap replicate by
src.evaluation.paired_bootstrap.paired_patient_bootstrap, called with its
defaults (sensitivity 0.95, 2000 replicates, seed 42).

Per-study predictions stay in authorised Drive storage; only the aggregate
table is written.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd

from src.evaluation.paired_bootstrap import paired_patient_bootstrap

PREFIX = "train/XR_WRIST/"
COLUMNS = ["patient_id", "study_id", "label", "abnormal_probability"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a-csv", type=Path, required=True)
    parser.add_argument("--model-b-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path, usecols=COLUMNS)
    rows["study_id"] = rows["study_id"].astype(str).str.removeprefix(PREFIX)
    rows["patient_id"] = rows["patient_id"].astype(str)
    return rows


def main() -> int:
    args = parse_args()
    model_a = load(args.model_a_csv)
    model_b = load(args.model_b_csv)
    pair = model_a.merge(model_b, on="study_id", suffixes=("_a", "_b"), validate="one_to_one")
    if not len(pair) == len(model_a) == len(model_b):
        raise ValueError("Model A and Model B are not evaluated on the same studies.")
    if not (pair["label_a"] == pair["label_b"]).all():
        raise ValueError("Labels differ between Model A and Model B.")
    if not (pair["patient_id_a"] == pair["patient_id_b"]).all():
        raise ValueError("Patient IDs differ between Model A and Model B.")

    result = paired_patient_bootstrap(
        pair["label_a"].to_numpy(),
        pair["abnormal_probability_a"].to_numpy(),
        pair["abnormal_probability_b"].to_numpy(),
        pair["patient_id_a"].to_numpy(),
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)

    print(f"studies={len(pair)} patients={pair['patient_id_a'].nunique()}")
    print(f"model_a_sha256={sha256(args.model_a_csv)}")
    print(f"model_b_sha256={sha256(args.model_b_csv)}")
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
