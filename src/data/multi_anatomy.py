"""Safe development manifests for all seven MURA anatomy categories.

This module deliberately does not alter the historical XR_WRIST manifest or
its results.  It creates a new development-only population from **official
MURA train** and rejects any path from official ``valid``.  In particular, the
historical MURA valid split is not reused for multi-anatomy model selection.

Multi-anatomy study IDs include the anatomy.  The legacy wrist IDs omit it,
which is safe only for an XR_WRIST-only project and would collide in a combined
MURA manifest.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import hashlib
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .index_mura import IMAGE_SUFFIXES, infer_label_from_study_name
from .split_patients import SplitRecord, validate_split_records, write_split_manifest


SUPPORTED_MURA_ANATOMIES: tuple[str, ...] = (
    "XR_ELBOW",
    "XR_FINGER",
    "XR_FOREARM",
    "XR_HAND",
    "XR_HUMERUS",
    "XR_SHOULDER",
    "XR_WRIST",
)

MULTI_ANATOMY_MANIFEST_COLUMNS: tuple[str, ...] = (
    "anatomy",
    "patient_id",
    "study_id",
    "image_path",
    "label",
    "source_split",
    "num_views_in_study",
)


@dataclass(frozen=True)
class MultiAnatomyRecord:
    """One real image view in the all-anatomy MURA train manifest."""

    anatomy: str
    patient_id: str
    study_id: str
    image_path: str
    label: int
    source_split: str
    num_views_in_study: int

    def __post_init__(self) -> None:
        if self.anatomy not in SUPPORTED_MURA_ANATOMIES:
            raise ValueError(f"Unsupported MURA anatomy: {self.anatomy!r}")
        if not self.patient_id or not self.study_id or not self.image_path:
            raise ValueError("anatomy, patient_id, study_id, and image_path must be non-empty")
        if self.label not in (0, 1):
            raise ValueError("label must be 0 or 1")
        if self.source_split != "train":
            raise ValueError("Multi-anatomy development accepts official MURA train only")
        if self.num_views_in_study < 1:
            raise ValueError("num_views_in_study must be positive")

    def as_dict(self) -> dict[str, str | int]:
        return {
            "anatomy": self.anatomy,
            "patient_id": self.patient_id,
            "study_id": self.study_id,
            "image_path": self.image_path,
            "label": self.label,
            "source_split": self.source_split,
            "num_views_in_study": self.num_views_in_study,
        }


@dataclass(frozen=True)
class ParsedMultiAnatomyPath:
    """Canonical parts inferred from an official MURA image path."""

    anatomy: str
    patient_id: str
    study_directory: str
    study_id: str
    label: int
    relative_image_path: str


def _normalise_path(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.strip().replace("\\", "/").split("/") if part not in {"", "."})


def parse_train_image_path(path: str, *, allowed_anatomies: Sequence[str] = SUPPORTED_MURA_ANATOMIES) -> ParsedMultiAnatomyPath:
    """Parse an official **train** image path and reject official-valid input.

    Expected source shape: ``MURA-v1.1/train/XR_*/patient*/study*_/image*.png``.
    ``study_id`` deliberately includes the anatomy to avoid cross-anatomy
    collisions such as the same patient's ``study1_positive`` directory.
    """

    parts = _normalise_path(path)
    try:
        train_index = parts.index("train")
    except ValueError as exc:
        if "valid" in parts:
            raise ValueError("Official MURA valid paths are forbidden in multi-anatomy development.") from exc
        raise ValueError(f"Expected an official MURA train image path: {path!r}") from exc
    if "valid" in parts:
        raise ValueError("Official MURA valid paths are forbidden in multi-anatomy development.")
    if len(parts) < train_index + 5:
        raise ValueError(f"Incomplete MURA train image path: {path!r}")
    anatomy, patient_id, study_directory, image_name = parts[train_index + 1 : train_index + 5]
    allowed = {value.upper() for value in allowed_anatomies}
    if anatomy.upper() not in allowed:
        raise ValueError(f"Unsupported or unrequested MURA anatomy in {path!r}")
    anatomy = anatomy.upper()
    if not patient_id.lower().startswith("patient"):
        raise ValueError(f"Could not identify patient directory in {path!r}")
    if not study_directory.lower().startswith("study"):
        raise ValueError(f"Could not identify study directory in {path!r}")
    if Path(image_name).suffix.lower() not in IMAGE_SUFFIXES or image_name.startswith("._"):
        raise ValueError(f"Not a supported MURA image file: {path!r}")
    label = infer_label_from_study_name(study_directory)
    if label is None:
        raise ValueError(f"Could not infer MURA normal/abnormal label from {path!r}")
    relative_path = "/".join(("train", anatomy, patient_id, study_directory, image_name))
    return ParsedMultiAnatomyPath(
        anatomy=anatomy,
        patient_id=patient_id,
        study_directory=study_directory,
        study_id=f"{anatomy}/{patient_id}/{study_directory}",
        label=int(label),
        relative_image_path=relative_path,
    )


def iter_train_image_paths(source_csv: str | Path) -> Iterator[str]:
    """Yield image paths from an official MURA train-image CSV.

    The official release has no header in this file; a defensive header check
    is retained because user-supplied copies sometimes add one.
    """

    path = Path(source_csv)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            if not row or not row[0].strip():
                continue
            value = row[0].strip()
            if row_number == 1 and value.lower() in {"image_path", "path", "filepath", "file_path"}:
                continue
            yield value


def build_train_manifest(
    source_csv: str | Path,
    *,
    allowed_anatomies: Sequence[str] = SUPPORTED_MURA_ANATOMIES,
) -> list[MultiAnatomyRecord]:
    """Build a deterministic combined image manifest from official train only."""

    allowed = tuple(value.upper() for value in allowed_anatomies)
    if not allowed or any(value not in SUPPORTED_MURA_ANATOMIES for value in allowed):
        raise ValueError("allowed_anatomies must be a non-empty subset of SUPPORTED_MURA_ANATOMIES")
    provisional: list[ParsedMultiAnatomyPath] = []
    for raw_path in iter_train_image_paths(source_csv):
        parts = _normalise_path(raw_path)
        if "valid" in parts:
            raise ValueError("The source CSV contains official valid paths; stop before manifest creation.")
        try:
            parsed = parse_train_image_path(raw_path, allowed_anatomies=allowed)
        except ValueError as exc:
            # Entries from an anatomy intentionally excluded from this build
            # are skipped, but malformed paths and valid paths remain errors.
            if "Unsupported or unrequested MURA anatomy" in str(exc):
                continue
            raise
        provisional.append(parsed)
    if not provisional:
        raise ValueError("No requested train images were found in source CSV.")

    counts: Counter[str] = Counter(item.study_id for item in provisional)
    records = [
        MultiAnatomyRecord(
            anatomy=item.anatomy,
            patient_id=item.patient_id,
            study_id=item.study_id,
            image_path=item.relative_image_path,
            label=item.label,
            source_split="train",
            num_views_in_study=counts[item.study_id],
        )
        for item in provisional
    ]
    records.sort(key=lambda item: (item.anatomy, item.patient_id, item.study_id, item.image_path))
    return records


def write_multi_anatomy_manifest(records: Iterable[MultiAnatomyRecord], output_path: str | Path) -> Path:
    """Write a portable combined manifest with an explicit anatomy column."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MULTI_ANATOMY_MANIFEST_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_dict())
    return output


def global_patient_split(
    records: Sequence[MultiAnatomyRecord], *, folds: int = 5, seed: int = 42
) -> list[SplitRecord]:
    """Make one patient-safe internal train/validation split across anatomies.

    ``StratifiedGroupKFold`` sees an anatomy×label stratum for every study,
    while the group is the original MURA patient ID.  This means a patient who
    appears in several body parts can never leak between train and validation.
    Fold 0 is selected deterministically; with five folds it approximates a
    80/20 development partition.
    """

    study_rows: dict[str, MultiAnatomyRecord] = {}
    for record in records:
        current = study_rows.setdefault(record.study_id, record)
        if (current.patient_id, current.anatomy, current.label) != (record.patient_id, record.anatomy, record.label):
            raise ValueError(f"Inconsistent study metadata for {record.study_id}")
    ordered = [study_rows[key] for key in sorted(study_rows)]
    if len({record.patient_id for record in ordered}) < folds:
        raise ValueError("Not enough unique patients for the requested number of grouped folds.")
    y = np.asarray([f"{record.anatomy}|{record.label}" for record in ordered], dtype=str)
    groups = np.asarray([record.patient_id for record in ordered], dtype=str)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=int(seed))
    _train_idx, val_idx = next(splitter.split(np.zeros(len(ordered)), y, groups))
    val_patients = set(groups[val_idx].tolist())
    splits = [
        SplitRecord(
            patient_id=record.patient_id,
            study_id=record.study_id,
            final_split="val" if record.patient_id in val_patients else "train",
        )
        for record in ordered
    ]
    validate_split_records(splits)
    return sorted(splits, key=lambda item: (item.patient_id, item.study_id))


def records_by_anatomy(records: Sequence[MultiAnatomyRecord]) -> dict[str, list[MultiAnatomyRecord]]:
    """Return deterministic per-anatomy manifest rows."""

    grouped: dict[str, list[MultiAnatomyRecord]] = defaultdict(list)
    for record in records:
        grouped[record.anatomy].append(record)
    return {anatomy: sorted(grouped.get(anatomy, []), key=lambda item: (item.patient_id, item.study_id, item.image_path)) for anatomy in SUPPORTED_MURA_ANATOMIES}


def split_records_by_anatomy(
    records: Sequence[MultiAnatomyRecord], splits: Sequence[SplitRecord]
) -> dict[str, list[SplitRecord]]:
    """Filter global assignments per anatomy without creating new splits."""

    anatomy_by_study = {record.study_id: record.anatomy for record in records}
    result: dict[str, list[SplitRecord]] = {anatomy: [] for anatomy in SUPPORTED_MURA_ANATOMIES}
    for split in splits:
        anatomy = anatomy_by_study.get(split.study_id)
        if anatomy is None:
            raise ValueError(f"Split contains a study missing from the combined manifest: {split.study_id}")
        result[anatomy].append(split)
    return {anatomy: sorted(rows, key=lambda item: (item.patient_id, item.study_id)) for anatomy, rows in result.items()}


def split_summary(records: Sequence[MultiAnatomyRecord], splits: Sequence[SplitRecord]) -> list[dict[str, int | str | float]]:
    """Calculate an aggregate-only audit table for each anatomy and split."""

    assignment = {record.study_id: record.final_split for record in splits}
    studies: dict[tuple[str, str], MultiAnatomyRecord] = {}
    for record in records:
        if record.study_id not in assignment:
            raise ValueError(f"Unassigned study: {record.study_id}")
        studies.setdefault(record.study_id, record)
    result: list[dict[str, int | str | float]] = []
    for anatomy in SUPPORTED_MURA_ANATOMIES:
        for final_split in ("train", "val"):
            study_rows = [record for record in studies.values() if record.anatomy == anatomy and assignment[record.study_id] == final_split]
            study_ids = {record.study_id for record in study_rows}
            image_count = sum(record.num_views_in_study for record in study_rows)
            labels = [record.label for record in study_rows]
            view_counts = [record.num_views_in_study for record in study_rows]
            result.append(
                {
                    "anatomy": anatomy,
                    "final_split": final_split,
                    "patients": len({record.patient_id for record in study_rows}),
                    "studies": len(study_ids),
                    "images": image_count,
                    "normal_studies": sum(label == 0 for label in labels),
                    "abnormal_studies": sum(label == 1 for label in labels),
                    "mean_views_per_study": round(float(np.mean(view_counts)), 3) if view_counts else 0.0,
                }
            )
    return result


def data_fingerprint(records: Sequence[MultiAnatomyRecord]) -> str:
    """Stable fingerprint over safe manifest rows for provenance records."""

    digest = hashlib.sha256()
    for record in records:
        digest.update("|".join(map(str, record.as_dict().values())).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


__all__ = [
    "MULTI_ANATOMY_MANIFEST_COLUMNS",
    "SUPPORTED_MURA_ANATOMIES",
    "MultiAnatomyRecord",
    "ParsedMultiAnatomyPath",
    "build_train_manifest",
    "data_fingerprint",
    "global_patient_split",
    "parse_train_image_path",
    "records_by_anatomy",
    "split_records_by_anatomy",
    "split_summary",
    "write_multi_anatomy_manifest",
    "write_split_manifest",
]
