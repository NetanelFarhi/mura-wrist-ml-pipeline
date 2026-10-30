#!/usr/bin/env python
"""Run integrity and geometry QC for the all-anatomy development manifests.

The script is intentionally a preprocessing audit, not a model-training
operation.  It reads only image paths already indexed from official MURA
train, verifies decodeability, and records aggregate geometry/QC facts by
anatomy and patient-safe internal split.  It never reads MURA valid.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.multi_anatomy import MULTI_ANATOMY_MANIFEST_COLUMNS, SUPPORTED_MURA_ANATOMIES
from src.data.split_patients import read_split_manifest, validate_split_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        missing = set(MULTI_ANATOMY_MANIFEST_COLUMNS) - set(fields)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        rows = [{key: str(value or "") for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError("Manifest is empty.")
    return rows


def inspect_image(row: dict[str, str], dataset_root: Path) -> dict[str, Any]:
    """Decode one image without materialising pixels in output artifacts."""

    relative_path = row["image_path"].replace("\\", "/")
    if relative_path.startswith("valid/") or "/valid/" in relative_path:
        return {"image_path": relative_path, "status": "FAIL", "issue": "official_valid_path_forbidden"}
    path = dataset_root / Path(relative_path)
    result: dict[str, Any] = {
        "image_path": relative_path,
        "status": "PASS",
        "issue": "",
        "width": None,
        "height": None,
        "mode": "",
    }
    if not path.is_file():
        result.update(status="FAIL", issue="missing_file")
        return result
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            result.update(width=int(width), height=int(height), mode=str(image.mode))
            # A narrow native view can still be a valid MURA radiograph.  The
            # fixed preprocessing transform handles it by padding/resizing;
            # retain it as a documented warning instead of silently dropping
            # a real study view from training or evaluation.
            if width < 128 or height < 128:
                result.update(status="WARNING", issue="below_128px_native_dimension")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        result.update(status="FAIL", issue=f"decode_error:{type(exc).__name__}")
    return result


def validate_manifest_hierarchy(rows: Sequence[dict[str, str]], split_rows: Sequence[Any]) -> None:
    assignments = {(item.patient_id, item.study_id): item.final_split for item in split_rows}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["anatomy"] not in SUPPORTED_MURA_ANATOMIES:
            raise ValueError(f"Unexpected anatomy: {row['anatomy']}")
        if row["source_split"] != "train":
            raise ValueError("Multi-anatomy QC may only read official train rows.")
        if not row["study_id"].startswith(f"{row['anatomy']}/"):
            raise ValueError(f"study_id must include anatomy: {row['study_id']}")
        if (row["patient_id"], row["study_id"]) not in assignments:
            raise ValueError(f"Study is missing a global split assignment: {row['study_id']}")
        grouped[row["study_id"]].append(row)
    for study_id, study_rows in grouped.items():
        anatomies = {row["anatomy"] for row in study_rows}
        labels = {row["label"] for row in study_rows}
        patients = {row["patient_id"] for row in study_rows}
        declared_counts = {int(row["num_views_in_study"]) for row in study_rows}
        paths = {row["image_path"] for row in study_rows}
        if len(anatomies) != 1 or len(labels) != 1 or len(patients) != 1:
            raise ValueError(f"Inconsistent anatomy/label/patient within {study_id}")
        if declared_counts != {len(paths)}:
            raise ValueError(f"Incorrect view count recorded for {study_id}")


def percentile(values: Iterable[float], q: float) -> float | None:
    array = np.asarray(list(values), dtype=float)
    return float(np.quantile(array, q)) if len(array) else None


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    rows = read_manifest(args.manifest)
    split_rows = read_split_manifest(args.split_manifest)
    validate_split_records(split_rows)
    validate_manifest_hierarchy(rows, split_rows)
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        inspections = list(executor.map(lambda row: inspect_image(row, dataset_root), rows))
    by_path = {item["image_path"]: item for item in inspections}
    assignments = {(item.patient_id, item.study_id): item.final_split for item in split_rows}

    study_rows: dict[str, dict[str, str]] = {}
    for row in rows:
        study_rows.setdefault(row["study_id"], row)
    summary_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    for anatomy in SUPPORTED_MURA_ANATOMIES:
        for final_split in ("train", "val"):
            selected_rows = [
                row for row in rows
                if row["anatomy"] == anatomy and assignments[(row["patient_id"], row["study_id"])] == final_split
            ]
            selected_studies = {
                row["study_id"]: row for row in selected_rows
            }
            selected_inspections = [by_path[row["image_path"]] for row in selected_rows]
            failures = [item for item in selected_inspections if item["status"] == "FAIL"]
            warnings = [item for item in selected_inspections if item["status"] == "WARNING"]
            labels = [int(row["label"]) for row in selected_studies.values()]
            summary_rows.append(
                {
                    "anatomy": anatomy,
                    "final_split": final_split,
                    "patients": len({row["patient_id"] for row in selected_studies.values()}),
                    "studies": len(selected_studies),
                    "images": len(selected_rows),
                    "normal_studies": labels.count(0),
                    "abnormal_studies": labels.count(1),
                    "decode_pass_images": len(selected_inspections) - len(failures) - len(warnings),
                    "decode_warning_images": len(warnings),
                    "decode_fail_images": len(failures),
                }
            )
            widths = [float(item["width"]) for item in selected_inspections if item["status"] != "FAIL"]
            heights = [float(item["height"]) for item in selected_inspections if item["status"] != "FAIL"]
            ratios = [width / height for width, height in zip(widths, heights) if height]
            geometry_rows.append(
                {
                    "anatomy": anatomy,
                    "final_split": final_split,
                    "images": len(selected_inspections),
                    "width_min": min(widths) if widths else None,
                    "width_max": max(widths) if widths else None,
                    "height_min": min(heights) if heights else None,
                    "height_max": max(heights) if heights else None,
                    "aspect_ratio_p05": percentile(ratios, 0.05),
                    "aspect_ratio_median": percentile(ratios, 0.50),
                    "aspect_ratio_p95": percentile(ratios, 0.95),
                    "source_modes": ";".join(f"{key}:{value}" for key, value in sorted(Counter(item["mode"] for item in selected_inspections if item["status"] != "FAIL").items())),
                }
            )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "qc_summary_by_anatomy.csv"
    geometry_path = output / "geometry_summary_by_anatomy.csv"
    issues_path = output / "qc_issues.csv"
    write_csv(summary_path, summary_rows, list(summary_rows[0]))
    write_csv(geometry_path, geometry_rows, list(geometry_rows[0]))
    issue_rows = [item for item in inspections if item["status"] != "PASS"]
    failure_rows = [item for item in inspections if item["status"] == "FAIL"]
    warning_rows = [item for item in inspections if item["status"] == "WARNING"]
    write_csv(issues_path, issue_rows, ("image_path", "status", "issue", "width", "height", "mode"))
    report = {
        "scope": "MURA official train only; all seven anatomies",
        "official_valid_read": False,
        "input_manifest": str(args.manifest.resolve()),
        "input_split_manifest": str(args.split_manifest.resolve()),
        "dataset_root": str(dataset_root),
        "total_images": len(rows),
        "total_studies": len(study_rows),
        "total_decode_failures": len(failure_rows),
        "total_low_resolution_warnings": len(warning_rows),
        "preprocessing_contract": {
            "input": "grayscale radiograph converted to three channels",
            "geometry": "aspect-ratio-preserving resize with square padding to 224x224",
            "evaluation": "deterministic; ImageNet normalization",
            "training": "augmentation is selected once in the model-training protocol, never during QC",
            "raw_pixel_storage": "none",
        },
    }
    report_path = output / "qc_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"summary": summary_path, "geometry": geometry_path, "issues": issues_path, "report": report_path}


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
