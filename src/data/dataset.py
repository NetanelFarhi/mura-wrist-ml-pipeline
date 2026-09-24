"""Image-level MURA dataset with explicit study/patient metadata.

Image-level loading is used for frozen-feature extraction.  It never changes
the prediction unit: callers can use ``study_id`` to mean-pool embeddings and
the companion :mod:`study_dataset` keeps all views together for evaluation.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .index_mura import MANIFEST_COLUMNS, ManifestRecord, read_manifest, resolve_image_path
from .split_patients import VALID_FINAL_SPLITS, join_manifest_with_splits, validate_manifest_studies

try:  # Torch is an application dependency, but importing data validation should stay lightweight.
    from torch.utils.data import Dataset as _Dataset
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal audit environments.
    class _Dataset:  # type: ignore[no-redef]
        pass


def _validate_row(row: Mapping[str, object]) -> dict[str, Any]:
    missing = [column for column in MANIFEST_COLUMNS if column not in row]
    if missing:
        raise ValueError(f"Dataset row is missing manifest columns: {', '.join(missing)}")
    # Reuse the invariant checks in the canonical record type while preserving
    # joined fields such as final_split and any user-side bookkeeping columns.
    canonical = ManifestRecord(
        patient_id=str(row["patient_id"]),
        study_id=str(row["study_id"]),
        image_path=str(row["image_path"]),
        label=int(row["label"]),
        source_split=str(row["source_split"]),
        num_views_in_study=int(row["num_views_in_study"]),
    )
    result: dict[str, Any] = dict(row)
    result.update(canonical.as_dict())
    if "final_split" in result and result["final_split"] not in (None, ""):
        final_split = str(result["final_split"]).strip().lower()
        if final_split not in VALID_FINAL_SPLITS:
            raise ValueError(f"Unknown final_split {result['final_split']!r}")
        result["final_split"] = final_split
    return result


def load_dataset_rows(
    manifest: str | os.PathLike[str] | Sequence[ManifestRecord | Mapping[str, object]],
) -> list[dict[str, Any]]:
    """Load a manifest while retaining optional joined columns such as final_split."""

    if isinstance(manifest, (str, os.PathLike)):
        path = Path(manifest)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Manifest has no header: {path}")
            missing = [column for column in MANIFEST_COLUMNS if column not in reader.fieldnames]
            if missing:
                raise ValueError(f"Manifest {path} is missing columns: {', '.join(missing)}")
            return [_validate_row(row) for row in reader]
    rows: list[dict[str, Any]] = []
    for item in manifest:
        if isinstance(item, ManifestRecord):
            rows.append(item.as_dict())
        else:
            rows.append(_validate_row(item))
    return rows


def _normalise_split_argument(split: str | None, final_split: str | None) -> str | None:
    if split is not None and final_split is not None and split.strip().lower() != final_split.strip().lower():
        raise ValueError("Pass either split or final_split, or pass the same value to both")
    wanted = split if split is not None else final_split
    if wanted is None:
        return None
    normalised = wanted.strip().lower()
    if normalised not in VALID_FINAL_SPLITS:
        raise ValueError(f"Unknown final split: {wanted!r}")
    return normalised


def prepare_dataset_rows(
    manifest: str | os.PathLike[str] | Sequence[ManifestRecord | Mapping[str, object]],
    *,
    split_manifest: str | os.PathLike[str] | Sequence[Mapping[str, object]] | None = None,
    split: str | None = None,
    final_split: str | None = None,
    verify_study_labels: bool = True,
) -> list[dict[str, Any]]:
    """Load, optionally join, validate, and filter rows for either dataset class."""

    rows = load_dataset_rows(manifest)
    if split_manifest is not None:
        # join_manifest_with_splits accepts mappings and validates that every
        # manifest study has exactly one patient-safe assignment.
        rows = [dict(row) for row in join_manifest_with_splits(rows, split_manifest)]
    if verify_study_labels:
        # The function accepts mapping rows and ignores extra joined columns.
        validate_manifest_studies(rows)

    wanted = _normalise_split_argument(split, final_split)
    if wanted is not None:
        if not all("final_split" in row and row["final_split"] not in (None, "") for row in rows):
            raise ValueError(
                "split filtering requires a joined final_split column or a split_manifest argument"
            )
        rows = [row for row in rows if str(row["final_split"]).strip().lower() == wanted]
    return sorted(rows, key=lambda row: (str(row["patient_id"]), str(row["study_id"]), str(row["image_path"])))


def load_grayscale_image(path: str | os.PathLike[str], *, dataset_root: str | os.PathLike[str] | None = None) -> Any:
    """Load a radiograph as an independent grayscale Pillow image.

    Pillow is intentionally imported at load time so indexing, split checking,
    and CSV-only smoke tests do not need a deep-learning environment.
    """

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on installation state.
        raise RuntimeError("Pillow is required to load MURA image pixels; install pillow>=10.") from exc

    resolved = resolve_image_path(path, dataset_root)
    try:
        with Image.open(resolved) as image:
            image.load()
            return image.convert("L").copy()
    except FileNotFoundError:
        raise FileNotFoundError(f"Manifest image is missing: {resolved}") from None
    except OSError as exc:
        raise OSError(f"Manifest image is corrupt or unreadable: {resolved}") from exc


class MURAImageDataset(_Dataset):
    """One-image samples with all patient/study identifiers retained.

    Parameters accept an exact six-column manifest, a CSV with an additional
    ``final_split`` column, or a sequence of record dictionaries.  ``split``
    filters against ``final_split``; it never filters against the original
    source split, avoiding accidental use of official validation data for fit.
    """

    def __init__(
        self,
        manifest: str | os.PathLike[str] | Sequence[ManifestRecord | Mapping[str, object]],
        *,
        dataset_root: str | os.PathLike[str] | None = None,
        split_manifest: str | os.PathLike[str] | Sequence[Mapping[str, object]] | None = None,
        split: str | None = None,
        final_split: str | None = None,
        transform: Any | None = None,
        verify_study_labels: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        self.transform = transform
        self.records = prepare_dataset_rows(
            manifest,
            split_manifest=split_manifest,
            split=split,
            final_split=final_split,
            verify_study_labels=verify_study_labels,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        image = load_grayscale_image(row["image_path"], dataset_root=self.dataset_root)
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": int(row["label"]),
            "patient_id": str(row["patient_id"]),
            "study_id": str(row["study_id"]),
            "image_path": str(row["image_path"]),
            "source_split": str(row["source_split"]),
            "final_split": row.get("final_split"),
            "num_views_in_study": int(row["num_views_in_study"]),
        }

