#!/usr/bin/env python
"""Create patient-safe all-anatomy MURA development manifests.

The command accepts only the official MURA *train* image-path CSV.  It builds
one global patient-level internal train/validation split, then writes a
combined manifest and seven anatomy-specific manifest/split pairs.  It never
opens MURA valid, so it cannot use the historically observed official-valid
population for any multi-anatomy development choice.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.multi_anatomy import (
    SUPPORTED_MURA_ANATOMIES,
    build_train_manifest,
    data_fingerprint,
    global_patient_split,
    records_by_anatomy,
    split_records_by_anatomy,
    split_summary,
    write_multi_anatomy_manifest,
)
from src.data.split_patients import write_split_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-image-paths", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5, help="Five folds yields approximately 80/20 train/validation.")
    return parser


def write_summary(rows: list[dict[str, int | str | float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    source = args.train_image_paths.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    # Names and source content are checked before any output is created.
    if "valid" in source.name.lower():
        raise ValueError("Pass the official train_image_paths CSV, never a valid CSV.")
    records = build_train_manifest(source)
    splits = global_patient_split(records, folds=args.folds, seed=args.seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    combined_manifest = write_multi_anatomy_manifest(records, output / "all_anatomies_train_manifest.csv")
    global_split = write_split_manifest(splits, output / "all_anatomies_global_split.csv")
    manifests_dir = output / "by_anatomy"
    by_anatomy = records_by_anatomy(records)
    anatomy_splits = split_records_by_anatomy(records, splits)
    for anatomy in SUPPORTED_MURA_ANATOMIES:
        stem = anatomy.lower()
        write_multi_anatomy_manifest(by_anatomy[anatomy], manifests_dir / f"{stem}_manifest.csv")
        write_split_manifest(anatomy_splits[anatomy], manifests_dir / f"{stem}_split.csv")

    summary = split_summary(records, splits)
    summary_path = output / "development_split_summary.csv"
    write_summary(summary, summary_path)
    provenance = {
        "scope": "MURA multi-anatomy development (official train only)",
        "supported_anatomies": list(SUPPORTED_MURA_ANATOMIES),
        "source_csv": str(source),
        "source_csv_sha256": __import__("hashlib").sha256(source.read_bytes()).hexdigest(),
        "manifest_sha256": data_fingerprint(records),
        "seed": int(args.seed),
        "folds": int(args.folds),
        "validation_fold": 0,
        "split_unit": "patient_id globally across all anatomies",
        "official_valid_read": False,
        "test_used_for_selection": False,
        "test_set_governance": "Official MURA valid is excluded from this extension; it was historically observed for XR_WRIST and is not an untouched advanced-model test.",
    }
    provenance_path = output / "development_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {
        "combined_manifest": combined_manifest,
        "global_split": global_split,
        "summary": summary_path,
        "provenance": provenance_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outputs = run(args)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
