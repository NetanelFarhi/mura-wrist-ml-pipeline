"""Reproducible study-level fine-tuning utilities for the Phase 1 ResNet-18.

The model produces one logit per radiograph.  This trainer converts logits to
probabilities, aggregates *within each padded study*, and computes a single
loss/prediction per study.  It never treats the views in a study as independent
train/test examples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import random
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.training.losses import StudyProbabilityBCELoss
from src.training.metrics import BinaryStudyMetrics, compute_binary_study_metrics, safe_auprc, safe_auroc


AggregationMethod = Literal["mean", "max"]
MonitorMetric = Literal["auroc", "auprc"]


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """Set Python, NumPy, and PyTorch seeds for a reproducible controlled run."""

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # warn_only permits a documented fallback for an unsupported operation
        # rather than failing after a long data-loading run.
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_data_loader_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from its PyTorch-assigned initial seed."""

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_data_loader_generator(seed: int) -> torch.Generator:
    """Return a generator suitable for DataLoader's ``generator=`` argument."""

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Choose CUDA when available, otherwise CPU; never silently fake CUDA."""

    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False. Use --device cpu.")
    return resolved


def _field(batch: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in batch:
            return batch[name]
    return None


def _as_list(value: Any, expected_length: int, name: str) -> list[Any]:
    if value is None:
        raise KeyError(f"Study batch is missing required field {name!r}.")
    if isinstance(value, Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, (str, bytes)):
        values = [value]
    else:
        values = list(value)
    if len(values) != expected_length:
        raise ValueError(f"Study batch field {name!r} has {len(values)} values, expected {expected_length}.")
    return values


def _pad_study_image_sequence(images: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    """Fallback for a custom collate function that leaves variable views as a list."""

    if not images:
        raise ValueError("A study batch cannot contain an empty image sequence.")
    normalized: list[Tensor] = []
    for study_images in images:
        if not isinstance(study_images, Tensor):
            raise TypeError("Each study image collection must be a torch Tensor.")
        if study_images.ndim == 3:
            study_images = study_images.unsqueeze(0)
        if study_images.ndim != 4:
            raise ValueError("Each study image tensor must have shape [views,C,H,W].")
        if study_images.shape[0] < 1:
            raise ValueError("Every study must contain at least one view.")
        normalized.append(study_images)
    reference_shape = tuple(normalized[0].shape[1:])
    if any(tuple(item.shape[1:]) != reference_shape for item in normalized):
        raise ValueError("All transformed views in a batch must share [C,H,W].")
    max_views = max(int(item.shape[0]) for item in normalized)
    padded = torch.zeros(
        (len(normalized), max_views, *reference_shape),
        dtype=normalized[0].dtype,
        device=normalized[0].device,
    )
    mask = torch.zeros((len(normalized), max_views), dtype=torch.bool, device=normalized[0].device)
    for index, item in enumerate(normalized):
        padded[index, : item.shape[0]] = item
        mask[index, : item.shape[0]] = True
    return padded, mask


def unpack_study_batch(
    batch: Mapping[str, Any],
) -> tuple[Tensor, Tensor, Tensor, list[str], list[str]]:
    """Normalize ``MURAStudyDataset`` batches to padded tensors and metadata.

    Supported primary format is ``images: [B,V,C,H,W]`` plus ``view_mask`` and
    study labels/IDs.  A list of variable-length view tensors is accepted as a
    defensive fallback.  A 4D image tensor is treated as a batch of one-view
    studies, useful for the CPU smoke test.
    """

    if not isinstance(batch, Mapping):
        raise TypeError("Expected a mapping batch from MURAStudyDataset/collate_studies.")
    images_value = _field(batch, "images", "image")
    if isinstance(images_value, Tensor):
        images = images_value
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5:
            raise ValueError(
                "Study images must have shape [batch, views, channels, height, width]; "
                f"received {tuple(images.shape)}."
            )
        batch_size, max_views = int(images.shape[0]), int(images.shape[1])
        mask_value = _field(batch, "view_mask", "mask", "views_mask")
        if mask_value is None:
            view_mask = torch.ones((batch_size, max_views), dtype=torch.bool, device=images.device)
        else:
            view_mask = torch.as_tensor(mask_value, dtype=torch.bool, device=images.device)
            if tuple(view_mask.shape) != (batch_size, max_views):
                raise ValueError(
                    "view_mask must have [batch, views] shape matching images; "
                    f"got {tuple(view_mask.shape)} and {(batch_size, max_views)}."
                )
    elif isinstance(images_value, Sequence):
        images, view_mask = _pad_study_image_sequence(images_value)
        batch_size = int(images.shape[0])
    else:
        raise TypeError("Study batch must contain Tensor/list field 'images' (or 'image').")

    labels = torch.as_tensor(_field(batch, "label", "labels"), dtype=torch.float32)
    if labels.numel() != batch_size:
        raise ValueError(f"Study labels has {labels.numel()} values, expected batch size {batch_size}.")
    labels = labels.reshape(-1)
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Study labels must be binary 0/1.")
    if torch.any(view_mask.sum(dim=1) < 1):
        raise ValueError("Every study batch row must have at least one unmasked view.")
    study_ids = [
        str(value)
        for value in _as_list(_field(batch, "study_id", "study_ids"), batch_size, "study_id")
    ]
    if any(not value for value in study_ids):
        raise ValueError("Every study must carry a non-empty study_id.")
    patient_value = _field(batch, "patient_id", "patient_ids")
    patient_ids = (
        [str(value) for value in _as_list(patient_value, batch_size, "patient_id")]
        if patient_value is not None
        else ["" for _ in range(batch_size)]
    )
    return images, view_mask, labels, study_ids, patient_ids


def aggregate_view_probabilities(
    view_probabilities: Tensor,
    view_mask: Tensor,
    *,
    method: AggregationMethod,
) -> Tensor:
    """Mean- or max-aggregate valid view probabilities into one per study."""

    if method not in ("mean", "max"):
        raise ValueError("Aggregation method must be 'mean' or 'max'.")
    if view_probabilities.ndim != 2 or view_mask.ndim != 2:
        raise ValueError("view_probabilities and view_mask must both have shape [batch, views].")
    if view_probabilities.shape != view_mask.shape:
        raise ValueError("view_probabilities and view_mask must share shape [batch, views].")
    mask = view_mask.bool()
    valid_counts = mask.sum(dim=1)
    if torch.any(valid_counts == 0):
        raise ValueError("Cannot aggregate a study with zero valid views.")
    if method == "mean":
        return (view_probabilities * mask.to(view_probabilities.dtype)).sum(dim=1) / valid_counts.to(
            view_probabilities.dtype
        )
    masked = view_probabilities.masked_fill(~mask, float("-inf"))
    return masked.max(dim=1).values


def _view_probabilities_for_padded_batch(
    model: nn.Module,
    images: Tensor,
    view_mask: Tensor,
) -> Tensor:
    """Run a view model only on real views, then restore [B,V] layout."""

    if images.ndim != 5:
        raise ValueError("images must be [batch, views, channels, height, width].")
    valid_images = images[view_mask]
    logits = model(valid_images).reshape(-1)
    if logits.numel() != valid_images.shape[0]:
        raise ValueError("Model must return exactly one logit per view.")
    probabilities = torch.zeros(
        view_mask.shape, dtype=logits.dtype, device=logits.device
    )
    probabilities[view_mask] = torch.sigmoid(logits)
    return probabilities


@dataclass
class StudyPredictions:
    """One prediction row per study, with both permitted aggregation variants."""

    labels: np.ndarray
    study_ids: list[str]
    patient_ids: list[str]
    num_views: np.ndarray
    probabilities_by_aggregation: dict[str, np.ndarray]

    def validate(self) -> None:
        n_studies = len(self.labels)
        if n_studies == 0:
            raise ValueError("Prediction set is empty.")
        if len(self.study_ids) != n_studies or len(self.patient_ids) != n_studies:
            raise ValueError("Prediction metadata does not match label rows.")
        if len(set(self.study_ids)) != n_studies:
            raise ValueError("Prediction loader emitted a study more than once.")
        if len(self.num_views) != n_studies or np.any(np.asarray(self.num_views) < 1):
            raise ValueError("Each prediction must have a positive number of views.")
        if not set(np.asarray(self.labels, dtype=int).tolist()).issubset({0, 1}):
            raise ValueError("Prediction labels must be binary 0/1.")
        for name, probabilities in self.probabilities_by_aggregation.items():
            if len(probabilities) != n_studies:
                raise ValueError(f"{name} probabilities do not match label rows.")
            if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
                raise ValueError(f"{name} probabilities must be finite values in [0, 1].")

    def rows(self, aggregation: AggregationMethod) -> list[dict[str, object]]:
        """Return CSV-ready rows for one selected aggregation method."""

        self.validate()
        if aggregation not in self.probabilities_by_aggregation:
            raise KeyError(f"No predictions stored for aggregation {aggregation!r}.")
        probabilities = self.probabilities_by_aggregation[aggregation]
        return [
            {
                "patient_id": self.patient_ids[index],
                "study_id": self.study_ids[index],
                "label": int(self.labels[index]),
                "num_views": int(self.num_views[index]),
                "abnormal_probability": float(probabilities[index]),
            }
            for index in range(len(self.labels))
        ]


@dataclass(frozen=True)
class AggregationSelection:
    """Validation-only decision between mean and max study aggregation."""

    selected_aggregation: AggregationMethod
    selection_metric: MonitorMetric
    validation_scores: dict[str, float]
    tie_break: str = "mean_preferred"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def select_aggregation_on_validation(
    validation_predictions: StudyPredictions,
    *,
    selection_metric: MonitorMetric = "auroc",
    candidates: Sequence[AggregationMethod] = ("mean", "max"),
) -> AggregationSelection:
    """Compare the required mean/max aggregations using validation only."""

    validation_predictions.validate()
    if selection_metric not in ("auroc", "auprc"):
        raise ValueError("selection_metric must be 'auroc' or 'auprc'.")
    if tuple(candidates) != ("mean", "max") and set(candidates) != {"mean", "max"}:
        raise ValueError("Phase 1 aggregation comparison must evaluate exactly mean and max.")
    scores: dict[str, float] = {}
    for method in ("mean", "max"):
        if method not in validation_predictions.probabilities_by_aggregation:
            raise ValueError(f"Validation predictions are missing required {method!r} aggregation.")
        probabilities = validation_predictions.probabilities_by_aggregation[method]
        score = (
            safe_auroc(validation_predictions.labels, probabilities)
            if selection_metric == "auroc"
            else safe_auprc(validation_predictions.labels, probabilities)
        )
        if score is None:
            raise ValueError("Validation aggregation selection requires both outcome classes.")
        scores[method] = float(score)
    # Mean has the deterministic tie-break, avoiding a data-free preference for
    # max when validation rank performance is identical.
    selected: AggregationMethod = "max" if scores["max"] > scores["mean"] + 1e-12 else "mean"
    return AggregationSelection(
        selected_aggregation=selected,
        selection_metric=selection_metric,
        validation_scores=scores,
    )


@dataclass(frozen=True)
class EpochSummary:
    epoch: int
    train_loss: float
    validation_monitor: float
    validation_metrics_at_0_5: dict[str, object]
    learning_rate: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class TrainingResult:
    """Outcome/provenance of an early-stopped training run."""

    best_epoch: int
    best_validation_monitor: float
    monitor_metric: MonitorMetric
    monitor_aggregation: AggregationMethod
    epochs_completed: int
    stopped_early: bool
    checkpoint_path: str
    history: list[EpochSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "best_epoch": self.best_epoch,
            "best_validation_monitor": self.best_validation_monitor,
            "monitor_metric": self.monitor_metric,
            "monitor_aggregation": self.monitor_aggregation,
            "epochs_completed": self.epochs_completed,
            "stopped_early": self.stopped_early,
            "checkpoint_path": self.checkpoint_path,
            "history": [entry.to_dict() for entry in self.history],
        }


class StudyLevelTrainer:
    """Fine-tune a view model while optimizing and evaluating at study level."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: StudyProbabilityBCELoss,
        *,
        scheduler: Any | None = None,
        device: str | torch.device | None = None,
        train_aggregation: AggregationMethod = "mean",
        gradient_clip_norm: float | None = None,
    ) -> None:
        if train_aggregation not in ("mean", "max"):
            raise ValueError("train_aggregation must be 'mean' or 'max'.")
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when supplied.")
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion.to(self.device)
        self.scheduler = scheduler
        self.train_aggregation = train_aggregation
        self.gradient_clip_norm = gradient_clip_norm

    def _study_probabilities(self, images: Tensor, view_mask: Tensor, aggregation: AggregationMethod) -> Tensor:
        view_probabilities = _view_probabilities_for_padded_batch(self.model, images, view_mask)
        return aggregate_view_probabilities(view_probabilities, view_mask, method=aggregation)

    def _train_epoch(self, train_loader: Iterable[Mapping[str, Any]]) -> float:
        self.model.train()
        total_loss = 0.0
        total_studies = 0
        for batch in train_loader:
            images, view_mask, labels, _, _ = unpack_study_batch(batch)
            images = images.to(self.device, non_blocking=True)
            view_mask = view_mask.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            probabilities = self._study_probabilities(images, view_mask, self.train_aggregation)
            loss = self.criterion(probabilities, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Encountered a non-finite training loss.")
            loss.backward()
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            self.optimizer.step()
            batch_size = int(labels.shape[0])
            total_loss += float(loss.detach().item()) * batch_size
            total_studies += batch_size
        if total_studies == 0:
            raise ValueError("Training DataLoader produced no studies.")
        return total_loss / total_studies

    @torch.inference_mode()
    def predict(
        self,
        data_loader: Iterable[Mapping[str, Any]],
        *,
        aggregations: Sequence[AggregationMethod] = ("mean", "max"),
    ) -> StudyPredictions:
        """Generate one probability per study for the supplied aggregation methods."""

        requested = tuple(aggregations)
        if not requested or any(method not in ("mean", "max") for method in requested):
            raise ValueError("aggregations must contain one or both of ('mean', 'max').")
        self.model.eval()
        labels_parts: list[np.ndarray] = []
        num_views_parts: list[np.ndarray] = []
        study_ids: list[str] = []
        patient_ids: list[str] = []
        probability_parts: dict[str, list[np.ndarray]] = {method: [] for method in requested}
        seen_study_ids: set[str] = set()
        for batch in data_loader:
            images, view_mask, labels, batch_study_ids, batch_patient_ids = unpack_study_batch(batch)
            duplicate = seen_study_ids.intersection(batch_study_ids)
            if duplicate:
                raise ValueError(
                    f"Study DataLoader repeated study IDs in evaluation: {sorted(duplicate)[:3]}"
                )
            seen_study_ids.update(batch_study_ids)
            images = images.to(self.device, non_blocking=True)
            view_mask = view_mask.to(self.device, non_blocking=True)
            view_probabilities = _view_probabilities_for_padded_batch(self.model, images, view_mask)
            for method in requested:
                probability_parts[method].append(
                    aggregate_view_probabilities(view_probabilities, view_mask, method=method)
                    .detach()
                    .cpu()
                    .numpy()
                )
            labels_parts.append(labels.detach().cpu().numpy().astype(np.int64, copy=False))
            num_views_parts.append(view_mask.sum(dim=1).detach().cpu().numpy().astype(np.int64, copy=False))
            study_ids.extend(batch_study_ids)
            patient_ids.extend(batch_patient_ids)
        if not labels_parts:
            raise ValueError("Evaluation DataLoader produced no studies.")
        result = StudyPredictions(
            labels=np.concatenate(labels_parts, axis=0),
            study_ids=study_ids,
            patient_ids=patient_ids,
            num_views=np.concatenate(num_views_parts, axis=0),
            probabilities_by_aggregation={
                method: np.concatenate(parts, axis=0).astype(np.float64, copy=False)
                for method, parts in probability_parts.items()
            },
        )
        result.validate()
        return result

    def evaluate(
        self,
        data_loader: Iterable[Mapping[str, Any]],
        *,
        aggregations: Sequence[AggregationMethod] = ("mean", "max"),
        threshold: float = 0.5,
    ) -> tuple[StudyPredictions, dict[str, BinaryStudyMetrics]]:
        """Predict and calculate fixed-threshold metrics without selecting anything."""

        predictions = self.predict(data_loader, aggregations=aggregations)
        metrics = {
            method: compute_binary_study_metrics(
                predictions.labels, predictions.probabilities_by_aggregation[method], threshold=threshold
            )
            for method in aggregations
        }
        return predictions, metrics

    def _checkpoint_payload(
        self,
        *,
        epoch: int,
        validation_monitor: float,
        validation_metrics: BinaryStudyMetrics,
        monitor_metric: MonitorMetric,
        monitor_aggregation: AggregationMethod,
    ) -> dict[str, Any]:
        provenance = self.model.provenance() if hasattr(self.model, "provenance") else {}
        return {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "validation_monitor": float(validation_monitor),
            "monitor_metric": monitor_metric,
            "monitor_aggregation": monitor_aggregation,
            "validation_metrics_at_0_5": validation_metrics.to_dict(),
            "train_aggregation": self.train_aggregation,
            "device_at_save": str(self.device),
            "model_provenance": provenance,
        }

    def save_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        epoch: int,
        validation_monitor: float,
        validation_metrics: BinaryStudyMetrics,
        monitor_metric: MonitorMetric,
        monitor_aggregation: AggregationMethod,
    ) -> Path:
        """Save a best-validation checkpoint with optimizer/scheduler state."""

        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._checkpoint_payload(
            epoch=epoch,
            validation_monitor=validation_monitor,
            validation_metrics=validation_metrics,
            monitor_metric=monitor_metric,
            monitor_aggregation=monitor_aggregation,
        )
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
        return path

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        load_optimizer: bool = False,
        load_scheduler: bool = False,
    ) -> dict[str, Any]:
        """Load a checkpoint; by default only restore model weights for evaluation."""

        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and checkpoint.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if load_scheduler and self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint

    def _step_scheduler(self, validation_monitor: float) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(validation_monitor)
        else:
            self.scheduler.step()

    def fit(
        self,
        train_loader: Iterable[Mapping[str, Any]],
        validation_loader: Iterable[Mapping[str, Any]],
        *,
        max_epochs: int,
        early_stopping_patience: int,
        checkpoint_path: str | Path,
        monitor_metric: MonitorMetric = "auroc",
        monitor_aggregation: AggregationMethod = "mean",
        min_delta: float = 0.0,
    ) -> TrainingResult:
        """Train with validation-only early stopping and best-checkpoint restore.

        The official/final test split is intentionally absent from this method's
        signature.  Use :meth:`predict` only after the model, aggregation, and
        operating threshold have been frozen from training/validation data.
        """

        if max_epochs < 1:
            raise ValueError("max_epochs must be at least 1.")
        if early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1.")
        if monitor_metric not in ("auroc", "auprc"):
            raise ValueError("monitor_metric must be 'auroc' or 'auprc'.")
        if monitor_aggregation not in ("mean", "max"):
            raise ValueError("monitor_aggregation must be 'mean' or 'max'.")
        if min_delta < 0:
            raise ValueError("min_delta cannot be negative.")

        history: list[EpochSummary] = []
        best_epoch = 0
        best_monitor = float("-inf")
        epochs_without_improvement = 0
        stopped_early = False
        resolved_checkpoint = Path(checkpoint_path)
        for epoch in range(1, max_epochs + 1):
            train_loss = self._train_epoch(train_loader)
            validation_predictions = self.predict(
                validation_loader, aggregations=(monitor_aggregation,)
            )
            validation_metrics = compute_binary_study_metrics(
                validation_predictions.labels,
                validation_predictions.probabilities_by_aggregation[monitor_aggregation],
                threshold=0.5,
            )
            monitor_value = (
                validation_metrics.auroc
                if monitor_metric == "auroc"
                else validation_metrics.auprc
            )
            if monitor_value is None:
                raise ValueError(
                    f"Validation {monitor_metric} is undefined because the validation split has one class."
                )
            monitor_value = float(monitor_value)
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            history.append(
                EpochSummary(
                    epoch=epoch,
                    train_loss=float(train_loss),
                    validation_monitor=monitor_value,
                    validation_metrics_at_0_5=validation_metrics.to_dict(),
                    learning_rate=learning_rate,
                )
            )
            if monitor_value > best_monitor + min_delta:
                best_epoch = epoch
                best_monitor = monitor_value
                epochs_without_improvement = 0
                self.save_checkpoint(
                    resolved_checkpoint,
                    epoch=epoch,
                    validation_monitor=monitor_value,
                    validation_metrics=validation_metrics,
                    monitor_metric=monitor_metric,
                    monitor_aggregation=monitor_aggregation,
                )
            else:
                epochs_without_improvement += 1
            self._step_scheduler(monitor_value)
            if epochs_without_improvement >= early_stopping_patience:
                stopped_early = True
                break

        if best_epoch == 0:
            raise RuntimeError("No best checkpoint was created during training.")
        # Subsequent aggregation/threshold selection always sees the
        # validation-best model, rather than the final overfit epoch.
        self.load_checkpoint(resolved_checkpoint)
        return TrainingResult(
            best_epoch=best_epoch,
            best_validation_monitor=float(best_monitor),
            monitor_metric=monitor_metric,
            monitor_aggregation=monitor_aggregation,
            epochs_completed=len(history),
            stopped_early=stopped_early,
            checkpoint_path=str(resolved_checkpoint),
            history=history,
        )


# A concise alias helps callers that expect a generic Trainer while preserving
# the explicit study-level name in documentation and saved code.
Trainer = StudyLevelTrainer


def save_json(payload: Mapping[str, Any] | TrainingResult, output_path: str | Path) -> Path:
    """Write a UTF-8, stable JSON artifact used by the command-line scripts."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable: Any = payload.to_dict() if isinstance(payload, TrainingResult) else payload
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


__all__ = [
    "AggregationMethod",
    "AggregationSelection",
    "EpochSummary",
    "MonitorMetric",
    "StudyLevelTrainer",
    "StudyPredictions",
    "Trainer",
    "TrainingResult",
    "aggregate_view_probabilities",
    "make_data_loader_generator",
    "resolve_device",
    "save_json",
    "seed_data_loader_worker",
    "select_aggregation_on_validation",
    "set_global_seed",
    "unpack_study_batch",
]
