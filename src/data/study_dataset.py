"""Study-level MURA dataset that keeps every view together."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dataset import load_grayscale_image, prepare_dataset_rows
from .index_mura import ManifestRecord

try:
    from torch.utils.data import Dataset as _Dataset
except ModuleNotFoundError:  # pragma: no cover - minimal CSV-only environments.
    class _Dataset:  # type: ignore[no-redef]
        pass


@dataclass(frozen=True)
class StudySample:
    """Metadata for a study; image pixels are loaded only in ``__getitem__``."""

    patient_id: str
    study_id: str
    label: int
    source_split: str
    final_split: str | None
    image_paths: tuple[str, ...]

    @property
    def num_views(self) -> int:
        return len(self.image_paths)


class MURAStudyDataset(_Dataset):
    """Return complete MURA studies rather than treating views as independent.

    Each item is a dictionary with an ``images`` list, one study label, and a
    deterministic ``image_paths`` ordering.  Use :func:`collate_studies` with a
    PyTorch DataLoader to produce a padded ``[batch, views, channels, height,
    width]`` tensor and a view mask.  No view is dropped by this class.
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
        rows = prepare_dataset_rows(
            manifest,
            split_manifest=split_manifest,
            split=split,
            final_split=final_split,
            verify_study_labels=verify_study_labels,
        )
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(str(row["patient_id"]), str(row["study_id"]))].append(row)

        self.studies: list[StudySample] = []
        for (patient_id, study_id), study_rows in sorted(grouped.items()):
            labels = {int(row["label"]) for row in study_rows}
            source_splits = {str(row["source_split"]) for row in study_rows}
            final_splits = {row.get("final_split") for row in study_rows}
            if len(labels) != 1:
                raise ValueError(f"Study {study_id!r} has inconsistent image labels: {sorted(labels)}")
            if len(source_splits) != 1:
                raise ValueError(f"Study {study_id!r} spans source splits: {sorted(source_splits)}")
            if len(final_splits) != 1:
                raise ValueError(
                    f"Study {study_id!r} spans final splits: "
                    f"{sorted(str(value) for value in final_splits)}"
                )
            image_paths = tuple(sorted(str(row["image_path"]) for row in study_rows))
            if not image_paths:
                raise ValueError(f"Study {study_id!r} contains no images")
            self.studies.append(
                StudySample(
                    patient_id=patient_id,
                    study_id=study_id,
                    label=next(iter(labels)),
                    source_split=next(iter(source_splits)),
                    final_split=next(iter(final_splits)),
                    image_paths=image_paths,
                )
            )

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, index: int) -> dict[str, Any]:
        study = self.studies[index]
        images: list[Any] = []
        for image_path in study.image_paths:
            image = load_grayscale_image(image_path, dataset_root=self.dataset_root)
            images.append(self.transform(image) if self.transform is not None else image)
        return {
            "images": images,
            "label": study.label,
            "patient_id": study.patient_id,
            "study_id": study.study_id,
            "image_paths": list(study.image_paths),
            "num_views": study.num_views,
            "source_split": study.source_split,
            "final_split": study.final_split,
        }
