"""Index the official MURA XR_WRIST layout into a portable image manifest.

The public MURA release stores labels in study-directory names (for example,
``study1_positive``) and also ships image/study CSV files.  This module uses
either source without making assumptions about counts or numeric identifiers.
It intentionally does *not* deduplicate input rows: duplicate paths are a
data-quality finding that the audit must be able to report.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Mapping, Sequence


MANIFEST_COLUMNS = (
    "patient_id",
    "study_id",
    "image_path",
    "label",
    "source_split",
    "num_views_in_study",
)

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
WRIST_ANATOMY = "XR_WRIST"


@dataclass(frozen=True)
class ManifestRecord:
    """One image row in the Phase 1 XR_WRIST manifest.

    ``study_id`` is deliberately patient-qualified (``patient/study``), so it
    remains globally unique even when different patients both have ``study1``.
    ``image_path`` is normally stored relative to the MURA root using POSIX
    separators, which makes a manifest portable between machines.
    """

    patient_id: str
    study_id: str
    image_path: str
    label: int
    source_split: str
    num_views_in_study: int

    def __post_init__(self) -> None:
        if not self.patient_id:
            raise ValueError("patient_id must not be empty")
        if not self.study_id:
            raise ValueError("study_id must not be empty")
        if not self.image_path:
            raise ValueError("image_path must not be empty")
        if int(self.label) not in (0, 1):
            raise ValueError(f"label must be 0 (normal) or 1 (abnormal), got {self.label!r}")
        if not self.source_split:
            raise ValueError("source_split must not be empty")
        if int(self.num_views_in_study) < 1:
            raise ValueError("num_views_in_study must be at least one")

    def as_dict(self) -> dict[str, str | int]:
        return {
            "patient_id": self.patient_id,
            "study_id": self.study_id,
            "image_path": self.image_path,
            "label": int(self.label),
            "source_split": self.source_split,
            "num_views_in_study": int(self.num_views_in_study),
        }


@dataclass(frozen=True)
class ParsedMURAPath:
    """The hierarchy and label inferred from a MURA path."""

    patient_id: str
    study_directory: str
    study_id: str
    inferred_label: int | None


def _normalise_path_text(value: str | os.PathLike[str]) -> str:
    """Return a slash-normalised, whitespace-free path representation."""

    text = str(value).strip().replace("\\", "/")
    # CSVs sometimes quote paths with a leading ``./``.  Keeping absolute paths
    # intact while removing that prefix makes path parsing predictable.
    while text.startswith("./"):
        text = text[2:]
    return text


def _path_segments(value: str | os.PathLike[str]) -> tuple[str, ...]:
    return tuple(segment for segment in _normalise_path_text(value).split("/") if segment not in ("", "."))


def infer_label_from_study_name(study_name: str) -> int | None:
    """Infer MURA's binary abnormality label from a study directory name."""

    name = study_name.strip().lower()
    if name.endswith("_positive"):
        return 1
    if name.endswith("_negative"):
        return 0
    return None


def parse_mura_path(path: str | os.PathLike[str]) -> ParsedMURAPath | None:
    """Parse a MURA XR_WRIST path, or return ``None`` for another anatomy.

    The function accepts a path to either an image or a study directory and is
    deliberately independent of the local filesystem.  That matters for audit
    runs, where a CSV can legitimately contain a missing file that still needs
    to be represented in the manifest.
    """

    segments = _path_segments(path)
    anatomy_index = next((i for i, item in enumerate(segments) if item.upper() == WRIST_ANATOMY), None)
    if anatomy_index is None:
        return None

    patient_index = next(
        (i for i in range(anatomy_index + 1, len(segments)) if segments[i].lower().startswith("patient")),
        None,
    )
    if patient_index is None:
        raise ValueError(f"Could not find a patient directory in MURA path: {path!s}")

    study_index = next(
        (i for i in range(patient_index + 1, len(segments)) if segments[i].lower().startswith("study")),
        None,
    )
    if study_index is None:
        raise ValueError(f"Could not find a study directory in MURA path: {path!s}")

    patient_id = segments[patient_index]
    study_directory = segments[study_index]
    return ParsedMURAPath(
        patient_id=patient_id,
        study_directory=study_directory,
        study_id=f"{patient_id}/{study_directory}",
        inferred_label=infer_label_from_study_name(study_directory),
    )


def _coerce_label(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "":
        return None
    if text in {"1", "1.0", "true", "positive", "abnormal"}:
        return 1
    if text in {"0", "0.0", "false", "negative", "normal"}:
        return 0
    raise ValueError(f"Unsupported MURA label value: {value!r}")


def _resolve_mura_path(dataset_root: Path, raw_path: str) -> Path:
    """Resolve a CSV path relative to the MURA root without changing its ID."""

    cleaned = _normalise_path_text(raw_path)
    candidate = Path(cleaned)
    if candidate.is_absolute():
        return candidate

    parts = _path_segments(cleaned)
    # Official CSV entries commonly begin with ``MURA-v1.1/`` even when the
    # caller already passed that directory as the dataset root.
    root_name = dataset_root.name
    if root_name in parts:
        parts = parts[parts.index(root_name) + 1 :]
    return dataset_root.joinpath(*parts)


def _portable_path(path: Path, dataset_root: Path) -> str:
    """Store paths relative to ``dataset_root`` where possible."""

    try:
        return path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except (ValueError, OSError):
        return str(path.resolve())


def resolve_image_path(image_path: str | os.PathLike[str], dataset_root: str | os.PathLike[str] | None = None) -> Path:
    """Resolve a manifest path, optionally against the MURA root.

    This is shared by datasets and auditing so a relative manifest means the
    same thing everywhere.  Relative paths require ``dataset_root``.
    """

    path = Path(_normalise_path_text(image_path))
    if path.is_absolute():
        return path
    if dataset_root is None:
        raise ValueError("dataset_root is required when manifest image_path values are relative")
    return Path(dataset_root).joinpath(path)


def _looks_like_image_path(raw_path: str) -> bool:
    return Path(_normalise_path_text(raw_path)).suffix.lower() in IMAGE_SUFFIXES


def _iter_study_images(study_dir: Path) -> Iterator[Path]:
    if not study_dir.is_dir():
        return
    yield from sorted(
        (path for path in study_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda item: item.as_posix(),
    )


def _read_csv_entries(csv_path: Path) -> Iterator[tuple[str, int | None]]:
    """Yield ``(path, optional_label)`` from supported official MURA CSVs."""

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return

    header = [cell.strip().lower() for cell in rows[0]]
    known_path_headers = {"image_path", "study_path", "path", "image", "filepath", "file_path"}
    known_label_headers = {"label", "target", "class"}
    has_header = bool(set(header) & (known_path_headers | known_label_headers))
    data_rows = rows[1:] if has_header else rows

    if has_header:
        path_index = next((i for i, name in enumerate(header) if name in known_path_headers), 0)
        label_index = next((i for i, name in enumerate(header) if name in known_label_headers), None)
    else:
        path_index, label_index = 0, 1 if len(rows[0]) > 1 else None

    for row_number, row in enumerate(data_rows, start=2 if has_header else 1):
        if not row or not any(cell.strip() for cell in row):
            continue
        if path_index >= len(row) or not row[path_index].strip():
            raise ValueError(f"CSV row {row_number} in {csv_path} has no path")
        label = _coerce_label(row[label_index]) if label_index is not None and label_index < len(row) else None
        yield row[path_index].strip(), label


def _source_directory_name(source_split: str) -> str:
    canonical = source_split.strip().lower()
    if canonical in {"valid", "validation", "official_valid", "test"}:
        return "valid"
    if canonical in {"train", "training"}:
        return "train"
    return source_split


def find_official_mura_csv(dataset_root: str | os.PathLike[str], source_split: str) -> Path | None:
    """Find an official source CSV, preferring image paths over study paths."""

    root = Path(dataset_root)
    source = _source_directory_name(source_split)
    candidates = (
        root / f"{source}_image_paths.csv",
        root / f"{source}_labeled_studies.csv",
        root / f"{source}_study_paths.csv",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _discover_wrist_images(dataset_root: Path, source_split: str) -> Iterator[tuple[Path, int | None]]:
    source_dir = dataset_root / _source_directory_name(source_split) / WRIST_ANATOMY
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Could not find {source_dir}. Expected an official MURA layout under {dataset_root}."
        )
    for image_path in sorted(
        (path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda item: item.as_posix(),
    ):
        yield image_path, None


def _entry_to_paths(dataset_root: Path, raw_path: str) -> tuple[Path, ...]:
    resolved = _resolve_mura_path(dataset_root, raw_path)
    if _looks_like_image_path(raw_path) or resolved.is_file():
        return (resolved,)
    if resolved.is_dir():
        return tuple(_iter_study_images(resolved))
    # A missing image still belongs in the manifest so the audit can report it.
    # A missing study directory cannot be represented as an image row.
    if _looks_like_image_path(raw_path):
        return (resolved,)
    return ()


def _as_manifest_record(raw: ManifestRecord | Mapping[str, object]) -> ManifestRecord:
    if isinstance(raw, ManifestRecord):
        return raw
    missing = [column for column in MANIFEST_COLUMNS if column not in raw]
    if missing:
        raise ValueError(f"Manifest row is missing columns: {', '.join(missing)}")
    return ManifestRecord(
        patient_id=str(raw["patient_id"]),
        study_id=str(raw["study_id"]),
        image_path=str(raw["image_path"]),
        label=int(raw["label"]),
        source_split=str(raw["source_split"]),
        num_views_in_study=int(raw["num_views_in_study"]),
    )


def build_wrist_manifest(
    dataset_root: str | os.PathLike[str],
    source_split: str,
    *,
    csv_path: str | os.PathLike[str] | None = None,
    output_csv: str | os.PathLike[str] | None = None,
    strict: bool = False,
) -> list[ManifestRecord]:
    """Index one official MURA source split for XR_WRIST.

    Parameters
    ----------
    dataset_root:
        Directory containing ``train/``, ``valid/``, and optionally the official
        MURA CSV files.
    source_split:
        Manifest provenance, normally ``"train"`` or ``"valid"``.  It is not
        changed to ``test`` here; split assignment is a separate, auditable step.
    csv_path:
        Optional image-path or labeled-study CSV.  Without it, the function uses
        the official CSV if present and otherwise walks ``XR_WRIST`` folders.
    strict:
        Raise on missing files, unlabelled studies, or a CSV/folder label conflict.
        With the default ``False``, missing image rows are retained for the audit.
    """

    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"MURA dataset root does not exist: {root}")

    chosen_csv = Path(csv_path) if csv_path is not None else find_official_mura_csv(root, source_split)
    if chosen_csv is not None and not chosen_csv.is_absolute() and not chosen_csv.is_file():
        rooted_csv = root / chosen_csv
        if rooted_csv.is_file():
            chosen_csv = rooted_csv
    if chosen_csv is not None and not chosen_csv.is_file():
        raise FileNotFoundError(f"MURA CSV does not exist: {chosen_csv}")

    source_entries: Iterable[tuple[Path | str, int | None]]
    if chosen_csv is not None:
        source_entries = _read_csv_entries(chosen_csv)
    else:
        source_entries = _discover_wrist_images(root, source_split)

    preliminary: list[tuple[ParsedMURAPath, Path, int]] = []
    for source_path, supplied_label in source_entries:
        raw_path = str(source_path)
        parsed = parse_mura_path(raw_path)
        if parsed is None:
            # Official CSVs cover all anatomies.  Silently skipping non-wrist
            # rows is intentional; malformed wrist-like rows still raise below.
            continue

        paths = (source_path,) if isinstance(source_path, Path) else _entry_to_paths(root, raw_path)
        if not paths:
            if strict:
                raise FileNotFoundError(f"No images found for MURA study path: {raw_path}")
            continue
        if parsed.inferred_label is None and supplied_label is None:
            raise ValueError(f"Cannot infer a normal/abnormal label from: {raw_path}")
        if (
            supplied_label is not None
            and parsed.inferred_label is not None
            and supplied_label != parsed.inferred_label
            and strict
        ):
            raise ValueError(
                f"CSV label {supplied_label} conflicts with study-folder label "
                f"{parsed.inferred_label} for {raw_path}"
            )
        label = supplied_label if supplied_label is not None else parsed.inferred_label
        assert label is not None  # established above; keeps type checking honest.
        for image_path in paths:
            if strict and not image_path.is_file():
                raise FileNotFoundError(f"Image listed by MURA metadata is missing: {image_path}")
            preliminary.append((parsed, image_path, label))

    # A study's view count represents distinct image paths, not duplicate CSV
    # rows.  Duplicates remain as rows and are surfaced later by audit_data.
    views_by_study: dict[tuple[str, str], set[str]] = {}
    for parsed, image_path, _ in preliminary:
        key = (parsed.patient_id, parsed.study_id)
        views_by_study.setdefault(key, set()).add(_portable_path(image_path, root))

    records = [
        ManifestRecord(
            patient_id=parsed.patient_id,
            study_id=parsed.study_id,
            image_path=_portable_path(image_path, root),
            label=label,
            source_split=source_split,
            num_views_in_study=len(views_by_study[(parsed.patient_id, parsed.study_id)]),
        )
        for parsed, image_path, label in preliminary
    ]
    records.sort(key=lambda item: (item.patient_id, item.study_id, item.image_path, item.label))
    if output_csv is not None:
        write_manifest(records, output_csv)
    return records


def build_combined_wrist_manifest(
    dataset_root: str | os.PathLike[str],
    *,
    source_splits: Sequence[str] = ("train", "valid"),
    output_csv: str | os.PathLike[str] | None = None,
    strict: bool = False,
) -> list[ManifestRecord]:
    """Index multiple source splits into one sorted XR_WRIST manifest."""

    records: list[ManifestRecord] = []
    for source_split in source_splits:
        records.extend(build_wrist_manifest(dataset_root, source_split, strict=strict))
    records.sort(key=lambda item: (item.source_split, item.patient_id, item.study_id, item.image_path))
    if output_csv is not None:
        write_manifest(records, output_csv)
    return records


def write_manifest(records: Iterable[ManifestRecord | Mapping[str, object]], output_csv: str | os.PathLike[str]) -> Path:
    """Write exactly the required manifest columns in a deterministic order."""

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalised = [_as_manifest_record(record) for record in records]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS), extrasaction="raise")
        writer.writeheader()
        for record in normalised:
            writer.writerow(record.as_dict())
    return output


def read_manifest(manifest: str | os.PathLike[str] | Sequence[ManifestRecord | Mapping[str, object]]) -> list[ManifestRecord]:
    """Read a manifest CSV or normalise in-memory manifest records."""

    if isinstance(manifest, (str, os.PathLike)):
        manifest_path = Path(manifest)
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Manifest has no header: {manifest_path}")
            missing = [column for column in MANIFEST_COLUMNS if column not in reader.fieldnames]
            if missing:
                raise ValueError(f"Manifest {manifest_path} is missing columns: {', '.join(missing)}")
            return [_as_manifest_record(row) for row in reader]
    return [_as_manifest_record(row) for row in manifest]


# A concise compatibility alias for callers that use the module name as a verb.
index_mura = build_wrist_manifest
