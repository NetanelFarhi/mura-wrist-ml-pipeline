"""Audit an XR_WRIST manifest before any model fitting.

Run from ``mura-x`` for example::

    python scripts/audit_data.py --data-root /data/MURA-v1.1 \
        --output reports/wrist_data_audit.json --fail-on-issues

The report contains measured counts only; it never substitutes expected MURA
counts when the dataset is unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# Permit direct ``python scripts/audit_data.py`` execution without installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_dataset_rows  # noqa: E402
from src.data.index_mura import (  # noqa: E402
    ManifestRecord,
    build_combined_wrist_manifest,
    parse_mura_path,
    resolve_image_path,
    write_manifest,
)
from src.data.split_patients import (  # noqa: E402
    SplitRecord,
    create_patient_splits,
    read_split_manifest,
    write_split_manifest,
)


def _png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    chunk_length = struct.unpack(">I", data[8:12])[0]
    if data[12:16] != b"IHDR" or chunk_length != 13:
        raise ValueError("PNG is missing a valid IHDR chunk")
    width, height = struct.unpack(">II", data[16:24])
    if width < 1 or height < 1:
        raise ValueError("PNG dimensions are invalid")
    if b"IEND" not in data:
        raise ValueError("PNG is truncated (no IEND chunk)")
    return width, height


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 4 or data[:2] != b"\xff\xd8" or data[-2:] != b"\xff\xd9":
        raise ValueError("not a complete JPEG file")
    index = 2
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while index < len(data) - 9:
        if data[index] != 0xFF:
            index += 1
            continue
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            break
        marker = data[index]
        index += 1
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > len(data):
            break
        segment_length = struct.unpack(">H", data[index : index + 2])[0]
        if segment_length < 2 or index + segment_length > len(data):
            raise ValueError("JPEG segment is truncated")
        if marker in sof_markers:
            if segment_length < 8:
                raise ValueError("JPEG SOF segment is invalid")
            height, width = struct.unpack(">HH", data[index + 3 : index + 7])
            if width < 1 or height < 1:
                raise ValueError("JPEG dimensions are invalid")
            return width, height
        index += segment_length
    raise ValueError("JPEG has no supported start-of-frame segment")


def _fallback_dimensions(path: Path) -> tuple[int, int]:
    """Read common image headers when Pillow is unavailable for a smoke audit."""

    data = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".png":
        return _png_dimensions(data)
    if suffix in {".jpg", ".jpeg"}:
        return _jpeg_dimensions(data)
    if suffix == ".bmp" and len(data) >= 26 and data[:2] == b"BM":
        width, height = struct.unpack("<ii", data[18:26])
        if width < 1 or height == 0:
            raise ValueError("BMP dimensions are invalid")
        return width, abs(height)
    raise ValueError(f"Pillow is unavailable and no safe header reader exists for {suffix or 'this file type'}")


def inspect_image(path: str | os.PathLike[str]) -> tuple[int, int]:
    """Verify an image and return ``(width, height)`` without model preprocessing."""

    image_path = Path(path)
    try:
        from PIL import Image
    except ModuleNotFoundError:
        return _fallback_dimensions(image_path)

    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
            if width < 1 or height < 1:
                raise OSError("image dimensions are invalid")
            return int(width), int(height)
    except (OSError, ValueError) as exc:
        raise ValueError(str(exc)) from exc


def _normalise_path_key(value: object) -> str:
    return os.path.normpath(str(value).replace("\\", "/")).replace("\\", "/")


def _distribution(values: Iterable[int]) -> dict[str, int]:
    return {str(value): count for value, count in sorted(Counter(values).items())}


def _split_overlap_report(
    rows: Sequence[Mapping[str, Any]],
    split_manifest: str | os.PathLike[str] | Sequence[SplitRecord | Mapping[str, object]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    assignments: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        final_split = row.get("final_split")
        if final_split not in (None, ""):
            assignments[(str(row["patient_id"]), str(row["study_id"]))].add(str(final_split).lower())
    if split_manifest is not None:
        for row in read_split_manifest(split_manifest):
            assignments[(row.patient_id, row.study_id)].add(row.final_split)

    patient_splits: dict[str, set[str]] = defaultdict(set)
    for (patient_id, _), splits in assignments.items():
        patient_splits[patient_id].update(splits)
    patient_overlap = [
        {"patient_id": patient_id, "final_splits": sorted(splits)}
        for patient_id, splits in sorted(patient_splits.items())
        if len(splits) > 1
    ]
    study_overlap = [
        {"patient_id": patient_id, "study_id": study_id, "final_splits": sorted(splits)}
        for (patient_id, study_id), splits in sorted(assignments.items())
        if len(splits) > 1
    ]
    unassigned = [
        {"patient_id": str(row["patient_id"]), "study_id": str(row["study_id"])}
        for row in rows
        if (str(row["patient_id"]), str(row["study_id"])) not in assignments
    ]
    # Repeated image rows otherwise repeat this same study diagnostic.
    unassigned = list({(item["patient_id"], item["study_id"]) for item in unassigned})
    return patient_overlap, study_overlap, [
        {"patient_id": patient_id, "study_id": study_id} for patient_id, study_id in sorted(unassigned)
    ]


def audit_manifest(
    manifest: str | os.PathLike[str] | Sequence[ManifestRecord | Mapping[str, object]],
    *,
    dataset_root: str | os.PathLike[str] | None = None,
    split_manifest: str | os.PathLike[str] | Sequence[SplitRecord | Mapping[str, object]] | None = None,
    inspect_pixels: bool = True,
) -> dict[str, Any]:
    """Return measured XR_WRIST quality, hierarchy, and split-integrity facts."""

    rows = load_dataset_rows(manifest)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["study_id"])].append(row)

    duplicate_counter = Counter(_normalise_path_key(row["image_path"]) for row in rows)
    duplicate_paths = [
        {"image_path": image_path, "count": count}
        for image_path, count in sorted(duplicate_counter.items())
        if count > 1
    ]
    inconsistent_labels: list[dict[str, Any]] = []
    study_patient_conflicts: list[dict[str, Any]] = []
    folder_label_mismatches: list[dict[str, Any]] = []
    study_labels: dict[str, int] = {}
    views_per_study: list[int] = []
    for study_id, study_rows in sorted(grouped.items()):
        labels = sorted({int(row["label"]) for row in study_rows})
        patients = sorted({str(row["patient_id"]) for row in study_rows})
        unique_paths = {_normalise_path_key(row["image_path"]) for row in study_rows}
        views_per_study.append(len(unique_paths))
        if len(labels) != 1:
            inconsistent_labels.append({"study_id": study_id, "labels": labels})
        else:
            study_labels[study_id] = labels[0]
        if len(patients) != 1:
            study_patient_conflicts.append({"study_id": study_id, "patient_ids": patients})
        for row in study_rows:
            try:
                parsed = parse_mura_path(str(row["image_path"]))
            except ValueError:
                parsed = None
            if parsed is not None and parsed.inferred_label is not None and parsed.inferred_label != int(row["label"]):
                folder_label_mismatches.append(
                    {
                        "patient_id": str(row["patient_id"]),
                        "study_id": study_id,
                        "image_path": str(row["image_path"]),
                        "manifest_label": int(row["label"]),
                        "folder_label": parsed.inferred_label,
                    }
                )

    missing_files: list[dict[str, str]] = []
    corrupt_images: list[dict[str, str]] = []
    dimensions: Counter[str] = Counter()
    widths: list[int] = []
    heights: list[int] = []
    if inspect_pixels:
        for row in rows:
            image_path = str(row["image_path"])
            try:
                resolved = resolve_image_path(image_path, dataset_root)
            except ValueError as exc:
                missing_files.append({"image_path": image_path, "error": str(exc)})
                continue
            if not resolved.is_file():
                missing_files.append({"image_path": image_path, "resolved_path": str(resolved)})
                continue
            try:
                width, height = inspect_image(resolved)
            except (OSError, ValueError) as exc:
                corrupt_images.append({"image_path": image_path, "resolved_path": str(resolved), "error": str(exc)})
                continue
            dimensions[f"{width}x{height}"] += 1
            widths.append(width)
            heights.append(height)

    patient_overlap, study_overlap, unassigned_studies = _split_overlap_report(rows, split_manifest)
    normal_studies = sum(1 for label in study_labels.values() if label == 0)
    abnormal_studies = sum(1 for label in study_labels.values() if label == 1)
    dimension_summary: dict[str, Any] = {
        "observed_image_count": sum(dimensions.values()),
        "by_width_x_height": dict(sorted(dimensions.items())),
    }
    if widths and heights:
        dimension_summary.update(
            {
                "width": {"min": min(widths), "max": max(widths)},
                "height": {"min": min(heights), "max": max(heights)},
            }
        )

    return {
        "scope": "XR_WRIST",
        "counts": {
            "patients": len({str(row["patient_id"]) for row in rows}),
            "studies": len(grouped),
            "images": len(rows),
            "unique_image_paths": len(duplicate_counter),
            "normal_studies": normal_studies,
            "abnormal_studies": abnormal_studies,
        },
        "views_per_study": {
            "distribution": _distribution(views_per_study),
            "min": min(views_per_study) if views_per_study else None,
            "max": max(views_per_study) if views_per_study else None,
            "mean": (sum(views_per_study) / len(views_per_study)) if views_per_study else None,
        },
        "image_dimensions": dimension_summary,
        "duplicate_paths": duplicate_paths,
        "missing_files": missing_files,
        "corrupt_images": corrupt_images,
        "inconsistent_labels_inside_study": inconsistent_labels,
        "study_patient_conflicts": study_patient_conflicts,
        "folder_label_mismatches": folder_label_mismatches,
        "patient_overlap_across_splits": patient_overlap,
        "study_overlap_across_splits": study_overlap,
        "unassigned_studies": unassigned_studies,
    }


def has_audit_issues(report: Mapping[str, Any]) -> bool:
    """Return whether issues should block fitting rather than merely be reported."""

    keys = (
        "duplicate_paths",
        "missing_files",
        "corrupt_images",
        "inconsistent_labels_inside_study",
        "study_patient_conflicts",
        "folder_label_mismatches",
        "patient_overlap_across_splits",
        "study_overlap_across_splits",
    )
    return any(bool(report.get(key)) for key in keys)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a MURA XR_WRIST manifest before Phase 1 fitting.")
    parser.add_argument("--manifest", type=Path, help="Existing wrist_manifest.csv (six columns, optional final_split).")
    parser.add_argument(
        "--dataset-root",
        "--data-root",
        dest="dataset_root",
        type=Path,
        help="MURA-v1.1 root; required for relative manifest paths.",
    )
    parser.add_argument("--split-manifest", type=Path, help="Optional patient-safe split_manifest.csv.")
    parser.add_argument(
        "--source-split",
        action="append",
        dest="source_splits",
        help="Source split to index if --manifest is omitted; repeat for train and valid.",
    )
    parser.add_argument("--output", type=Path, help="Write JSON report to this file; otherwise print it.")
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "wrist_manifest.csv",
        help="Where to write the indexed manifest when --manifest is omitted.",
    )
    parser.add_argument(
        "--split-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "split_manifest.csv",
        help="Where to write patient-safe assignments when --manifest is omitted.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help="Fraction of official training patients reserved for validation (default: 0.20).",
    )
    parser.add_argument("--seed", type=int, default=23, help="Deterministic patient-split seed (default: 23).")
    parser.add_argument("--skip-pixel-inspection", action="store_true", help="Only audit manifest and split identities.")
    parser.add_argument("--fail-on-issues", action="store_true", help="Return non-zero if fitting-blocking issues are found.")
    args = parser.parse_args(argv)
    if args.manifest is None and args.dataset_root is None:
        parser.error("one of --manifest (with absolute image paths) or --dataset-root is required")
    if args.manifest is not None and (args.manifest_output != PROJECT_ROOT / "data" / "manifests" / "wrist_manifest.csv" or args.split_output != PROJECT_ROOT / "data" / "manifests" / "split_manifest.csv"):
        parser.error("--manifest-output and --split-output are only valid when --manifest is omitted")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    generated_manifest: Path | None = None
    generated_split: Path | None = None
    try:
        if args.manifest is not None:
            source: Any = args.manifest
        else:
            source = build_combined_wrist_manifest(
                args.dataset_root,
                source_splits=tuple(args.source_splits or ("train", "valid")),
            )
            generated_manifest = write_manifest(source, args.manifest_output)
            generated_split = write_split_manifest(
                create_patient_splits(
                    source,
                    validation_fraction=args.validation_fraction,
                    seed=args.seed,
                    official_valid_as_test=True,
                ),
                args.split_output,
            )
        report = audit_manifest(
            source,
            dataset_root=args.dataset_root,
            split_manifest=args.split_manifest or generated_split,
            inspect_pixels=not args.skip_pixel_inspection,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if generated_manifest is not None and generated_split is not None:
        report["generated_artifacts"] = {
            "manifest": str(generated_manifest),
            "split_manifest": str(generated_split),
            "official_valid_is_sealed_test": True,
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
        }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 2 if args.fail_on_issues and has_audit_issues(report) else 0


if __name__ == "__main__":
    raise SystemExit(main())
