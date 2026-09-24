"""Collators that preserve MURA study boundaries and ragged view counts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _require_nonempty_batch(batch: Sequence[Mapping[str, Any]]) -> None:
    if not batch:
        raise ValueError("Cannot collate an empty batch")


def _metadata(batch: Sequence[Mapping[str, Any]], key: str) -> list[Any]:
    return [sample.get(key) for sample in batch]


def collate_images(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate image-level samples without discarding study identifiers."""

    _require_nonempty_batch(batch)
    images = [sample["image"] for sample in batch]
    labels = [int(sample["label"]) for sample in batch]
    try:
        import torch
    except ModuleNotFoundError:
        torch = None  # type: ignore[assignment]

    if torch is not None and all(isinstance(image, torch.Tensor) for image in images):
        image_batch: Any = torch.stack(images, dim=0)
        label_batch: Any = torch.tensor(labels, dtype=torch.long)
    else:
        image_batch = images
        label_batch = labels
    return {
        "images": image_batch,
        "image": image_batch,  # Alias keeps simple image-level trainers ergonomic.
        "labels": label_batch,
        "label": label_batch,
        "patient_ids": _metadata(batch, "patient_id"),
        "study_ids": _metadata(batch, "study_id"),
        "image_paths": _metadata(batch, "image_path"),
        "source_splits": _metadata(batch, "source_split"),
        "final_splits": _metadata(batch, "final_split"),
    }


def collate_studies(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad views within a batch and return a boolean mask for real images.

    If transforms returned tensors, ``images`` has shape
    ``[batch, max_views, channels, height, width]`` and ``view_mask`` has shape
    ``[batch, max_views]``.  The mask is mandatory for mean/max aggregation so
    padding cannot influence a study prediction.  In a no-torch inspection
    environment the same structure is returned as nested lists.
    """

    _require_nonempty_batch(batch)
    view_lists: list[list[Any]] = []
    for position, sample in enumerate(batch):
        images = list(sample.get("images", []))
        if not images:
            raise ValueError(f"Study batch item {position} has no images")
        view_lists.append(images)
    labels = [int(sample["label"]) for sample in batch]
    num_views = [len(images) for images in view_lists]
    max_views = max(num_views)

    try:
        import torch
    except ModuleNotFoundError:
        torch = None  # type: ignore[assignment]

    all_tensors = torch is not None and all(
        isinstance(image, torch.Tensor) for images in view_lists for image in images
    )
    if all_tensors:
        assert torch is not None
        exemplar = view_lists[0][0]
        if exemplar.ndim != 3:
            raise ValueError("Study tensors must have shape [channels, height, width]")
        expected_shape = tuple(exemplar.shape)
        expected_dtype = exemplar.dtype
        expected_device = exemplar.device
        for images in view_lists:
            for image in images:
                if tuple(image.shape) != expected_shape:
                    raise ValueError(
                        "All transformed views in a batch must share [channels, height, width]; "
                        f"got {tuple(image.shape)} and {expected_shape}"
                    )
                if image.dtype != expected_dtype or image.device != expected_device:
                    raise ValueError("All transformed views in a batch must share dtype and device")
        image_batch: Any = torch.zeros(
            (len(batch), max_views, *expected_shape), dtype=expected_dtype, device=expected_device
        )
        view_mask: Any = torch.zeros((len(batch), max_views), dtype=torch.bool, device=expected_device)
        for batch_index, images in enumerate(view_lists):
            for view_index, image in enumerate(images):
                image_batch[batch_index, view_index] = image
                view_mask[batch_index, view_index] = True
        label_batch: Any = torch.tensor(labels, dtype=torch.long, device=expected_device)
    else:
        image_batch = view_lists
        view_mask = [[view_index < count for view_index in range(max_views)] for count in num_views]
        label_batch = labels

    return {
        "images": image_batch,
        "labels": label_batch,
        "label": label_batch,
        "view_mask": view_mask,
        "num_views": num_views,
        "patient_ids": _metadata(batch, "patient_id"),
        "study_ids": _metadata(batch, "study_id"),
        "image_paths": _metadata(batch, "image_paths"),
        "source_splits": _metadata(batch, "source_split"),
        "final_splits": _metadata(batch, "final_split"),
    }


collate_study_batch = collate_studies

