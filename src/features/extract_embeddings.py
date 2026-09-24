"""Frozen ImageNet ResNet-18 embedding extraction for Model A.

Embeddings are generated per image and only then mean-pooled within a study.
The extractor never averages across studies or derives labels from directory
ordering.  It is intentionally independent of a particular DataLoader class:
it accepts the mapping batches emitted by ``MURAImageDataset`` and validates
the patient/study metadata needed for reproducible artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from src.models.resnet18_classifier import (
    ResNet18BuildInfo,
    WeightMode,
    build_resnet18_backbone,
)


@dataclass
class StudyEmbeddings:
    """Mean-pooled, one-row-per-study embeddings and their provenance."""

    embeddings: np.ndarray
    labels: np.ndarray
    study_ids: np.ndarray
    patient_ids: np.ndarray
    num_views: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate shape, label, and hierarchy invariants before persistence."""

        if self.embeddings.ndim != 2 or self.embeddings.shape[0] == 0:
            raise ValueError("embeddings must be a non-empty two-dimensional array.")
        n_studies = self.embeddings.shape[0]
        for name, value in (
            ("labels", self.labels),
            ("study_ids", self.study_ids),
            ("patient_ids", self.patient_ids),
            ("num_views", self.num_views),
        ):
            if len(value) != n_studies:
                raise ValueError(f"{name} has length {len(value)}, expected {n_studies}.")
        unique_labels = set(np.asarray(self.labels, dtype=int).tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Expected binary labels 0/1, found {sorted(unique_labels)}.")
        if len(set(map(str, self.study_ids))) != n_studies:
            raise ValueError("Study embeddings must contain exactly one row per study_id.")
        if np.any(np.asarray(self.num_views, dtype=int) < 1):
            raise ValueError("Every study must contribute at least one image embedding.")

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-safe summary without duplicating the embedding matrix."""

        self.validate()
        return {
            "n_studies": int(self.embeddings.shape[0]),
            "embedding_dim": self.embedding_dim,
            "n_normal": int(np.sum(self.labels == 0)),
            "n_abnormal": int(np.sum(self.labels == 1)),
            "mean_views_per_study": float(np.mean(self.num_views)),
            **self.metadata,
        }


class FrozenResNet18Embedder(nn.Module):
    """A ResNet-18 whose ImageNet backbone cannot accidentally be fine-tuned."""

    def __init__(
        self,
        *,
        weights: str | WeightMode = "imagenet",
        allow_weight_download: bool = False,
    ) -> None:
        super().__init__()
        backbone, build_info = build_resnet18_backbone(
            weights=weights, allow_weight_download=allow_weight_download
        )
        if not hasattr(backbone, "fc") or not isinstance(backbone.fc, nn.Linear):
            raise TypeError("Expected ResNet-18 backbone with a linear fc layer.")
        self.embedding_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.build_info: ResNet18BuildInfo = build_info
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "FrozenResNet18Embedder":
        """Keep BatchNorm and all other backbone layers in evaluation mode.

        This override protects the baseline from silently updating BatchNorm
        running statistics if a caller uses a generic training loop.
        """

        super().train(False)
        self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(
                "Expected image tensor [batch, channels, height, width], "
                f"received shape {tuple(images.shape)}."
            )
        features = self.backbone(images)
        return features.reshape(features.shape[0], -1)

    def provenance(self) -> dict[str, Any]:
        return {
            "model": "frozen_resnet18_imagenet_embedding",
            "embedding_dim": self.embedding_dim,
            "frozen": True,
            **self.build_info.to_dict(),
        }


def _as_python_list(value: Any, expected_length: int, name: str) -> list[Any]:
    """Normalize default-collated tensors/arrays/sequences to a list."""

    if value is None:
        raise KeyError(f"Batch is missing required field {name!r}.")
    if isinstance(value, Tensor):
        if value.ndim == 0:
            values = [value.item()]
        else:
            values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, (str, bytes)):
        values = [value]
    else:
        values = list(value)
    if len(values) != expected_length:
        raise ValueError(
            f"Batch field {name!r} has {len(values)} values, expected {expected_length}."
        )
    return values


def _field(batch: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in batch:
            return batch[name]
    return None


def _unpack_image_batch(
    batch: Mapping[str, Any],
) -> tuple[Tensor, list[int], list[str], list[str]]:
    """Extract individual views and matching metadata from an image batch.

    ``MURAImageDataset`` returns ``image``.  The 5D branch also supports a
    padded study batch, which is useful for small debugging runs but is not
    required for the standard extraction script.
    """

    images = _field(batch, "image", "images")
    if not isinstance(images, Tensor):
        raise TypeError("Embedding batches must contain an image tensor under 'image' or 'images'.")

    if images.ndim == 4:
        count = int(images.shape[0])
        labels = [int(x) for x in _as_python_list(_field(batch, "label", "labels"), count, "label")]
        study_ids = [str(x) for x in _as_python_list(_field(batch, "study_id", "study_ids"), count, "study_id")]
        patient_values = _field(batch, "patient_id", "patient_ids")
        patient_ids = (
            [str(x) for x in _as_python_list(patient_values, count, "patient_id")]
            if patient_values is not None
            else ["" for _ in range(count)]
        )
        return images, labels, study_ids, patient_ids

    if images.ndim != 5:
        raise ValueError(
            "Expected image batch [B,C,H,W] or padded study batch [B,V,C,H,W], "
            f"received {tuple(images.shape)}."
        )

    batch_size, max_views = int(images.shape[0]), int(images.shape[1])
    mask = _field(batch, "view_mask", "mask")
    if mask is None:
        view_mask = torch.ones((batch_size, max_views), dtype=torch.bool)
    elif isinstance(mask, Tensor):
        if tuple(mask.shape) != (batch_size, max_views):
            raise ValueError("view_mask must have shape [batch, views].")
        view_mask = mask.detach().cpu().bool()
    else:
        view_mask = torch.as_tensor(mask, dtype=torch.bool)
        if tuple(view_mask.shape) != (batch_size, max_views):
            raise ValueError("view_mask must have shape [batch, views].")

    study_labels = [
        int(x) for x in _as_python_list(_field(batch, "label", "labels"), batch_size, "label")
    ]
    study_ids_by_row = [
        str(x) for x in _as_python_list(_field(batch, "study_id", "study_ids"), batch_size, "study_id")
    ]
    patient_values = _field(batch, "patient_id", "patient_ids")
    patient_ids_by_row = (
        [str(x) for x in _as_python_list(patient_values, batch_size, "patient_id")]
        if patient_values is not None
        else ["" for _ in range(batch_size)]
    )

    flat_images: list[Tensor] = []
    flat_labels: list[int] = []
    flat_study_ids: list[str] = []
    flat_patient_ids: list[str] = []
    for row in range(batch_size):
        for view in torch.nonzero(view_mask[row], as_tuple=False).reshape(-1).tolist():
            flat_images.append(images[row, view])
            flat_labels.append(study_labels[row])
            flat_study_ids.append(study_ids_by_row[row])
            flat_patient_ids.append(patient_ids_by_row[row])
    if not flat_images:
        raise ValueError("Received a study batch with no valid views.")
    return torch.stack(flat_images), flat_labels, flat_study_ids, flat_patient_ids


def extract_study_embeddings(
    dataloader: Iterable[Mapping[str, Any]],
    embedder: FrozenResNet18Embedder,
    *,
    device: torch.device | str | None = None,
) -> StudyEmbeddings:
    """Extract frozen features and mean-pool all image embeddings per study.

    The DataLoader should be deterministic (``shuffle=False``) for convenient
    debugging.  Output order is nevertheless sorted by ``study_id`` so the
    saved matrix is reproducible even when batches arrive in a different order.
    """

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    embedder = embedder.to(resolved_device)
    embedder.eval()

    features_by_study: dict[str, list[np.ndarray]] = {}
    labels_by_study: dict[str, int] = {}
    patients_by_study: dict[str, str] = {}

    with torch.inference_mode():
        for batch in dataloader:
            if not isinstance(batch, Mapping):
                raise TypeError(
                    "Expected a mapping batch from MURAImageDataset; use its default DataLoader collation."
                )
            images, labels, study_ids, patient_ids = _unpack_image_batch(batch)
            vectors = embedder(images.to(resolved_device, non_blocking=True)).detach().cpu().numpy()
            if not (len(vectors) == len(labels) == len(study_ids) == len(patient_ids)):
                raise RuntimeError("Embedding batch outputs no longer align with its study metadata.")
            for vector, label, study_id, patient_id in zip(vectors, labels, study_ids, patient_ids):
                if label not in (0, 1):
                    raise ValueError(f"Study {study_id!r} has non-binary label {label!r}.")
                if not study_id:
                    raise ValueError("An image is missing study_id; cannot safely pool embeddings.")
                previous_label = labels_by_study.setdefault(study_id, label)
                if previous_label != label:
                    raise ValueError(
                        f"Inconsistent labels within study {study_id!r}: "
                        f"{previous_label} and {label}."
                    )
                if patient_id:
                    previous_patient = patients_by_study.setdefault(study_id, patient_id)
                    if previous_patient != patient_id:
                        raise ValueError(
                            f"Study {study_id!r} maps to multiple patient IDs: "
                            f"{previous_patient!r}, {patient_id!r}."
                        )
                else:
                    patients_by_study.setdefault(study_id, "")
                features_by_study.setdefault(study_id, []).append(vector.astype(np.float32, copy=False))

    if not features_by_study:
        raise ValueError("No images were supplied for embedding extraction.")

    ordered_study_ids = sorted(features_by_study)
    embeddings = np.stack(
        [np.mean(np.stack(features_by_study[study_id], axis=0), axis=0) for study_id in ordered_study_ids],
        axis=0,
    ).astype(np.float32, copy=False)
    result = StudyEmbeddings(
        embeddings=embeddings,
        labels=np.asarray([labels_by_study[study_id] for study_id in ordered_study_ids], dtype=np.int64),
        study_ids=np.asarray(ordered_study_ids, dtype=str),
        patient_ids=np.asarray([patients_by_study[study_id] for study_id in ordered_study_ids], dtype=str),
        num_views=np.asarray(
            [len(features_by_study[study_id]) for study_id in ordered_study_ids], dtype=np.int64
        ),
        metadata={
            "pooling": "mean_embedding",
            "device": str(resolved_device),
            **embedder.provenance(),
        },
    )
    result.validate()
    return result


def save_study_embeddings(embeddings: StudyEmbeddings, output_path: str | Path) -> Path:
    """Persist a portable compressed NPZ artifact with embedded JSON metadata."""

    embeddings.validate()
    path = Path(output_path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Embedding output path must end in .npz.")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = embeddings.to_metadata()
    np.savez_compressed(
        path,
        embeddings=np.asarray(embeddings.embeddings, dtype=np.float32),
        labels=np.asarray(embeddings.labels, dtype=np.int64),
        study_ids=np.asarray(embeddings.study_ids, dtype=str),
        patient_ids=np.asarray(embeddings.patient_ids, dtype=str),
        num_views=np.asarray(embeddings.num_views, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return path


def load_study_embeddings(input_path: str | Path) -> StudyEmbeddings:
    """Load and validate a Model A embedding artifact produced by this module."""

    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Embedding artifact does not exist: {path}")
    with np.load(path, allow_pickle=False) as artifact:
        required = {"embeddings", "labels", "study_ids", "patient_ids", "num_views", "metadata_json"}
        missing = required.difference(artifact.files)
        if missing:
            raise ValueError(f"Embedding artifact is missing required arrays: {sorted(missing)}")
        raw_metadata = artifact["metadata_json"].item()
        result = StudyEmbeddings(
            embeddings=np.asarray(artifact["embeddings"], dtype=np.float32),
            labels=np.asarray(artifact["labels"], dtype=np.int64),
            study_ids=np.asarray(artifact["study_ids"], dtype=str),
            patient_ids=np.asarray(artifact["patient_ids"], dtype=str),
            num_views=np.asarray(artifact["num_views"], dtype=np.int64),
            metadata=json.loads(str(raw_metadata)),
        )
    result.validate()
    return result


__all__ = [
    "FrozenResNet18Embedder",
    "StudyEmbeddings",
    "extract_study_embeddings",
    "load_study_embeddings",
    "save_study_embeddings",
]
