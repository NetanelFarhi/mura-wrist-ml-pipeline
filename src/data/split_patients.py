"""Deterministic, patient-safe train/validation/test split utilities.

The official MURA ``valid`` source is a sealed final test set by default.  A
validation set is therefore created only from official training patients.  This
module never samples individual images or studies independently.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .index_mura import MANIFEST_COLUMNS, ManifestRecord, read_manifest


SPLIT_COLUMNS = ("patient_id", "study_id", "final_split")
VALID_FINAL_SPLITS = frozenset({"train", "val", "test"})


@dataclass(frozen=True)
class SplitRecord:
    """One study-level assignment in the reproducible split manifest."""

    patient_id: str
    study_id: str
    final_split: str

    def __post_init__(self) -> None:
        if not self.patient_id or not self.study_id:
            raise ValueError("patient_id and study_id must not be empty")
        if self.final_split not in VALID_FINAL_SPLITS:
            raise ValueError(f"final_split must be one of {sorted(VALID_FINAL_SPLITS)}, got {self.final_split!r}")

    def as_dict(self) -> dict[str, str]:
        return {
            "patient_id": self.patient_id,
            "study_id": self.study_id,
            "final_split": self.final_split,
        }


@dataclass(frozen=True)
class StudySummary:
    """Validated study-level metadata derived from image-level manifest rows."""

    patient_id: str
    study_id: str
    label: int
    source_split: str
    image_paths: tuple[str, ...]


def _as_split_record(raw: SplitRecord | Mapping[str, object]) -> SplitRecord:
    if isinstance(raw, SplitRecord):
        return raw
    missing = [column for column in SPLIT_COLUMNS if column not in raw]
    if missing:
        raise ValueError(f"Split row is missing columns: {', '.join(missing)}")
    return SplitRecord(
        patient_id=str(raw["patient_id"]),
        study_id=str(raw["study_id"]),
        final_split=str(raw["final_split"]).strip().lower(),
    )


def read_split_manifest(
    split_manifest: str | os.PathLike[str] | Sequence[SplitRecord | Mapping[str, object]],
) -> list[SplitRecord]:
    """Read an exact three-column split manifest or normalise in-memory rows."""

    if isinstance(split_manifest, (str, os.PathLike)):
        path = Path(split_manifest)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Split manifest has no header: {path}")
            missing = [column for column in SPLIT_COLUMNS if column not in reader.fieldnames]
            if missing:
                raise ValueError(f"Split manifest {path} is missing columns: {', '.join(missing)}")
            return [_as_split_record(row) for row in reader]
    return [_as_split_record(row) for row in split_manifest]


def write_split_manifest(
    records: Iterable[SplitRecord | Mapping[str, object]], output_csv: str | os.PathLike[str]
) -> Path:
    """Write a study-level split manifest using only its specified columns."""

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [_as_split_record(record) for record in records]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SPLIT_COLUMNS), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    return output


def validate_manifest_studies(
    manifest: str | os.PathLike[str] | Sequence[ManifestRecord | Mapping[str, object]],
    *,
    validate_view_counts: bool = True,
) -> list[StudySummary]:
    """Validate that every image in a study agrees on identity and label.

    This is deliberately called before assigning any split.  A bad manifest is
    safer to reject than to silently turn into a leakage-prone experiment.
    """

    rows = read_manifest(manifest)
    if not rows:
        raise ValueError("Cannot create a split from an empty manifest")

    grouped: dict[str, list[ManifestRecord]] = defaultdict(list)
    for row in rows:
        grouped[row.study_id].append(row)

    summaries: list[StudySummary] = []
    for study_id, study_rows in grouped.items():
        patient_ids = {row.patient_id for row in study_rows}
        labels = {int(row.label) for row in study_rows}
        source_splits = {row.source_split.strip().lower() for row in study_rows}
        paths = [row.image_path for row in study_rows]
        if len(patient_ids) != 1:
            raise ValueError(f"Study {study_id!r} is assigned to multiple patients: {sorted(patient_ids)}")
        if len(labels) != 1:
            raise ValueError(f"Study {study_id!r} has inconsistent labels: {sorted(labels)}")
        if len(source_splits) != 1:
            raise ValueError(f"Study {study_id!r} spans source splits: {sorted(source_splits)}")
        if validate_view_counts:
            declared = {int(row.num_views_in_study) for row in study_rows}
            unique_path_count = len(set(paths))
            if len(declared) != 1 or next(iter(declared)) != unique_path_count:
                raise ValueError(
                    f"Study {study_id!r} declares {sorted(declared)} views but contains "
                    f"{unique_path_count} distinct image paths"
                )
        summaries.append(
            StudySummary(
                patient_id=next(iter(patient_ids)),
                study_id=study_id,
                label=next(iter(labels)),
                source_split=next(iter(source_splits)),
                image_paths=tuple(sorted(set(paths))),
            )
        )
    return sorted(summaries, key=lambda item: (item.patient_id, item.study_id))


def _stable_order(values: Iterable[str], *, seed: int, namespace: str) -> list[str]:
    """Order IDs reproducibly without relying on Python's salted hash()."""

    def sort_key(value: str) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}|{namespace}|{value}".encode("utf-8")).hexdigest()
        return digest, value

    return sorted(values, key=sort_key)


def _patient_label_signature(studies: Sequence[StudySummary]) -> str:
    """Represent whether a patient has normal, abnormal, or mixed studies."""

    labels = sorted({study.label for study in studies})
    return "|".join(str(label) for label in labels)


def _choose_validation_patients(
    train_patients: Mapping[str, Sequence[StudySummary]],
    *,
    validation_fraction: float,
    seed: int,
    stratify: bool,
) -> set[str]:
    """Choose patient IDs while retaining deterministic, lightweight stratification."""

    patient_ids = sorted(train_patients)
    total = len(patient_ids)
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if validation_fraction == 0.0 or total < 2:
        return set()

    target = int(round(total * validation_fraction))
    target = min(total - 1, max(1, target))
    if not stratify:
        return set(_stable_order(patient_ids, seed=seed, namespace="unstratified")[:target])

    strata: dict[str, list[str]] = defaultdict(list)
    for patient_id, studies in train_patients.items():
        strata[_patient_label_signature(studies)].append(patient_id)

    # Allocate close to the requested fraction in each patient-label stratum,
    # preserving at least one training patient per stratum whenever possible.
    allocation: dict[str, int] = {}
    capacity: dict[str, int] = {}
    remainder: dict[str, float] = {}
    for signature, ids in strata.items():
        desired = len(ids) * validation_fraction
        capacity[signature] = max(0, len(ids) - 1)
        allocation[signature] = min(math.floor(desired), capacity[signature])
        remainder[signature] = desired - math.floor(desired)

    allocated = sum(allocation.values())
    ranked_strata = sorted(strata, key=lambda key: (-remainder[key], key))
    while allocated < target:
        eligible = [key for key in ranked_strata if allocation[key] < capacity[key]]
        if not eligible:
            break
        for signature in eligible:
            if allocated >= target:
                break
            allocation[signature] += 1
            allocated += 1

    # If all strata are singletons, preserving every stratum would leave no
    # validation set.  Fall back deterministically while still retaining at
    # least one training patient globally.
    if allocated < target:
        already_selected: set[str] = set()
        for signature, count in allocation.items():
            already_selected.update(_stable_order(strata[signature], seed=seed, namespace=signature)[:count])
        candidates = _stable_order(
            (patient_id for patient_id in patient_ids if patient_id not in already_selected),
            seed=seed,
            namespace="singleton-fallback",
        )
        return already_selected | set(candidates[: target - allocated])

    selected: set[str] = set()
    for signature, ids in strata.items():
        selected.update(_stable_order(ids, seed=seed, namespace=signature)[: allocation[signature]])
    return selected


def create_patient_splits(
    manifest: str | os.PathLike[str] | Sequence[ManifestRecord | Mapping[str, object]],
    *,
    validation_fraction: float = 0.20,
    seed: int = 42,
    official_valid_as_test: bool = True,
    train_source_splits: Sequence[str] = ("train",),
    sealed_source_splits: Sequence[str] = ("test",),
    stratify: bool = True,
    validate_view_counts: bool = True,
) -> list[SplitRecord]:
    """Create deterministic patient-level train/val/test study assignments.

    By default, every MURA ``valid``/``official_valid`` study becomes ``test``
    and is never considered in validation selection.  To use a non-standard
    source as trainable data, callers must explicitly include it in
    ``train_source_splits``; this makes a potentially consequential decision
    visible in version-controlled code/configuration.
    """

    studies = validate_manifest_studies(manifest, validate_view_counts=validate_view_counts)
    train_sources = {source.strip().lower() for source in train_source_splits}
    sealed_sources = {source.strip().lower() for source in sealed_source_splits}
    if official_valid_as_test:
        sealed_sources.update({"valid", "validation", "official_valid"})

    if train_sources & sealed_sources:
        overlap = sorted(train_sources & sealed_sources)
        raise ValueError(f"A source split cannot be both trainable and sealed: {overlap}")

    patient_sources: dict[str, set[str]] = defaultdict(set)
    for study in studies:
        patient_sources[study.patient_id].add(study.source_split)
    source_leakage = {patient: values for patient, values in patient_sources.items() if len(values) > 1}
    if source_leakage:
        details = "; ".join(f"{patient}: {sorted(values)}" for patient, values in sorted(source_leakage.items()))
        raise ValueError(f"Patients overlap source splits; cannot create a safe split ({details})")

    train_patients: dict[str, list[StudySummary]] = defaultdict(list)
    test_patient_ids: set[str] = set()
    for study in studies:
        source = study.source_split
        if source in sealed_sources:
            test_patient_ids.add(study.patient_id)
        elif source in train_sources:
            train_patients[study.patient_id].append(study)
        else:
            raise ValueError(
                f"Unexpected source_split {study.source_split!r}. Add it explicitly to "
                "train_source_splits or sealed_source_splits."
            )

    if set(train_patients) & test_patient_ids:
        overlap = sorted(set(train_patients) & test_patient_ids)
        raise ValueError(f"Patients occur in both trainable and sealed sources: {overlap}")

    val_patient_ids = _choose_validation_patients(
        train_patients,
        validation_fraction=validation_fraction,
        seed=seed,
        stratify=stratify,
    )
    result: list[SplitRecord] = []
    for study in studies:
        if study.patient_id in test_patient_ids:
            final_split = "test"
        elif study.patient_id in val_patient_ids:
            final_split = "val"
        else:
            final_split = "train"
        result.append(SplitRecord(study.patient_id, study.study_id, final_split))

    validate_split_records(result)
    return sorted(result, key=lambda item: (item.patient_id, item.study_id))


def validate_split_records(
    split_manifest: str | os.PathLike[str] | Sequence[SplitRecord | Mapping[str, object]],
) -> None:
    """Raise if a split manifest leaks a patient or study across final splits."""

    rows = read_split_manifest(split_manifest)
    patient_assignments: dict[str, set[str]] = defaultdict(set)
    study_assignments: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        patient_assignments[row.patient_id].add(row.final_split)
        study_assignments[(row.patient_id, row.study_id)].add(row.final_split)
    leaking_patients = {key: values for key, values in patient_assignments.items() if len(values) > 1}
    leaking_studies = {key: values for key, values in study_assignments.items() if len(values) > 1}
    if leaking_patients:
        raise ValueError(f"Patient overlap across final splits: {leaking_patients}")
    if leaking_studies:
        raise ValueError(f"Study overlap across final splits: {leaking_studies}")
    study_keys = [(row.patient_id, row.study_id) for row in rows]
    if len(study_keys) != len(set(study_keys)):
        raise ValueError("Split manifest contains duplicate study assignments")


def join_manifest_with_splits(
    manifest: str | os.PathLike[str] | Sequence[ManifestRecord | Mapping[str, object]],
    split_manifest: str | os.PathLike[str] | Sequence[SplitRecord | Mapping[str, object]],
    *,
    require_complete: bool = True,
) -> list[dict[str, str | int]]:
    """Attach ``final_split`` to each image row without altering CSV schemas.

    The returned records are convenient inputs for datasets and training code;
    the persisted source manifest remains the exact six-column contract.
    """

    rows = read_manifest(manifest)
    split_rows = read_split_manifest(split_manifest)
    validate_split_records(split_rows)
    assignments = {(row.patient_id, row.study_id): row.final_split for row in split_rows}
    joined: list[dict[str, str | int]] = []
    missing: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.patient_id, row.study_id)
        final_split = assignments.get(key)
        if final_split is None:
            missing.add(key)
            if require_complete:
                continue
        item = row.as_dict()
        if final_split is not None:
            item["final_split"] = final_split
        joined.append(item)
    if missing and require_complete:
        formatted = ", ".join(f"{patient}/{study}" for patient, study in sorted(missing))
        raise ValueError(f"Split manifest does not assign every manifest study: {formatted}")
    return joined


def filter_manifest_by_split(
    records: Sequence[Mapping[str, object]], split: str
) -> list[dict[str, object]]:
    """Filter already joined manifest rows by their ``final_split`` value."""

    wanted = split.strip().lower()
    if wanted not in VALID_FINAL_SPLITS:
        raise ValueError(f"Unknown final split: {split!r}")
    return [dict(row) for row in records if str(row.get("final_split", "")).strip().lower() == wanted]
