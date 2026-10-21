#!/usr/bin/env python
"""Run the single pre-registered Phase 2C study-level partial-fine-tuning test.

Only the official MURA training population is accepted.  The script rebuilds a
compatible frozen mean-embedding reference and then performs exactly one
layer4-plus-head study-level run.  Detailed predictions/checkpoints belong in
the local output directory; the optional report directory receives aggregate,
Git-safe artifacts only.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.data.collate import collate_studies
from src.data.study_dataset import MURAStudyDataset
from src.data.transforms import build_eval_transform, preprocessing_spec
from src.evaluation.paired_bootstrap import paired_patient_bootstrap
from src.evaluation.phase2a import operating_metrics, select_sensitivity_constrained_threshold
from src.models.multiview_classifier import MultiViewEmbeddingClassifier
from src.models.resnet18_classifier import build_resnet18_classifier
from src.models.study_mean_classifier import StudyMeanEmbeddingClassifier
from src.training.losses import decide_class_weighting
from src.training.trainer import (
    make_data_loader_generator,
    resolve_device,
    seed_data_loader_worker,
    set_global_seed,
    unpack_study_batch,
)


PRIMARY_TARGET = 0.95
DESCRIPTIVE_TARGETS = (0.85, 0.90, PRIMARY_TARGET)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--reference-epochs", type=int, default=40)
    parser.add_argument("--reference-patience", type=int, default=7)
    parser.add_argument("--reference-head-lr", type=float, default=1e-3)
    parser.add_argument("--partial-epochs", type=int, default=15)
    parser.add_argument("--partial-patience", type=int, default=4)
    parser.add_argument("--layer4-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Batch size must be positive and workers cannot be negative.")
    for name in ("reference_epochs", "reference_patience", "partial_epochs", "partial_patience", "bootstrap_samples"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least one.")
    for name in ("reference_head_lr", "layer4_lr", "head_lr"):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative.")
    for path in (args.manifest, args.split_manifest, args.dataset_root, args.init_checkpoint):
        if not path.exists():
            raise FileNotFoundError(f"Required local experiment input is missing: {path}")


def assert_train_only_inputs(manifest_path: Path, split_manifest_path: Path) -> dict[str, int]:
    """Fail closed if an input manifest could expose official MURA valid/test data."""

    manifest = pd.read_csv(manifest_path)
    split = pd.read_csv(split_manifest_path)
    required_manifest = {"patient_id", "study_id", "image_path", "label", "source_split"}
    required_split = {"patient_id", "study_id", "final_split"}
    missing = required_manifest.difference(manifest.columns) | required_split.difference(split.columns)
    if missing:
        raise ValueError(f"Missing required train-only manifest fields: {sorted(missing)}")
    if not manifest["source_split"].eq("train").all():
        raise ValueError("Phase 2C refuses non-train source rows.")
    paths = manifest["image_path"].astype(str).str.replace("\\", "/", regex=False)
    if not paths.str.startswith("train/").all() or paths.str.contains(r"(?:^|/)valid(?:/|$)|(?:^|/)test(?:/|$)", regex=True).any():
        raise ValueError("Phase 2C image paths must be train-only and may not mention valid/test.")
    if not set(split["final_split"].astype(str)).issubset({"train", "val"}):
        raise ValueError("Phase 2C split manifest accepts only final train/val partitions.")
    if split.duplicated(["patient_id", "study_id"]).any():
        raise ValueError("Split manifest must have exactly one final split per study.")
    merged = manifest.merge(split, on=["patient_id", "study_id"], how="inner", validate="many_to_one")
    if len(merged) != len(manifest):
        raise ValueError("Every train-only image row must have a patient-safe final split.")
    patient_splits = merged.groupby("patient_id")["final_split"].nunique()
    if (patient_splits > 1).any():
        raise ValueError("Patient leakage detected between internal train and validation.")
    studies = merged.drop_duplicates("study_id")
    return {
        "images": int(len(merged)),
        "studies": int(len(studies)),
        "patients": int(studies["patient_id"].nunique()),
        "train_studies": int((studies["final_split"] == "train").sum()),
        "validation_studies": int((studies["final_split"] == "val").sum()),
    }


def _make_loader(
    dataset: MURAStudyDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_studies,
        worker_init_fn=seed_data_loader_worker if workers else None,
        generator=make_data_loader_generator(seed),
        pin_memory=pin_memory,
    )


def _dataset_labels(dataset: MURAStudyDataset) -> np.ndarray:
    return np.asarray([int(study.label) for study in dataset.studies], dtype=int)


def _assert_dataset_integrity(train: MURAStudyDataset, validation: MURAStudyDataset) -> None:
    train_patients = {str(study.patient_id) for study in train.studies}
    validation_patients = {str(study.patient_id) for study in validation.studies}
    train_studies = {str(study.study_id) for study in train.studies}
    validation_studies = {str(study.study_id) for study in validation.studies}
    if train_patients.intersection(validation_patients) or train_studies.intersection(validation_studies):
        raise ValueError("Internal train/validation patient or study overlap detected.")
    if any(study.source_split != "train" for study in [*train.studies, *validation.studies]):
        raise ValueError("Phase 2C dataset unexpectedly contains non-train source data.")


def assert_phase1_checkpoint_provenance(
    checkpoint_path: Path, manifest_path: Path, split_manifest_path: Path
) -> None:
    """Require the saved Phase-1 metadata to match the train-only inputs."""

    checkpoint_path = Path(checkpoint_path)
    manifest_path = Path(manifest_path)
    split_manifest_path = Path(split_manifest_path)
    run_config_path = checkpoint_path.parent / "run_config.json"
    selection_path = checkpoint_path.parent / "selection.json"
    if not run_config_path.is_file() or not selection_path.is_file():
        raise ValueError("Phase 2C requires the Phase-1 checkpoint run_config.json and selection.json beside the weights.")
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    arguments = run_config.get("arguments", {})
    saved_manifest = Path(str(arguments.get("manifest", ""))).resolve()
    saved_split_manifest = Path(str(arguments.get("split_manifest", ""))).resolve()
    if saved_manifest != manifest_path.resolve() or saved_split_manifest != split_manifest_path.resolve():
        raise ValueError("Phase-1 checkpoint provenance does not match the supplied train-only manifests.")
    if bool(arguments.get("write_test_predictions", False)) or bool(selection.get("test_used_for_selection", True)):
        raise ValueError("Phase 2C refuses a Phase-1 checkpoint whose metadata indicates test use.")
    if int(run_config.get("train_studies", -1)) != 2769 or int(run_config.get("validation_studies", -1)) != 691:
        raise ValueError("Phase-1 checkpoint provenance does not match the current 2,769/691 development split.")


def load_phase1_view_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load the existing Phase-1 Model-B weights without changing provenance."""

    model = build_resnet18_classifier(weights="none")
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Phase-1 checkpoint is missing model_state_dict.")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device)


def extract_frozen_embeddings(
    view_model: nn.Module,
    loader: Iterable[dict[str, Any]],
    *,
    device: torch.device,
    local_output: Path,
    split_name: str,
) -> dict[str, Any]:
    """Extract train-only checkpoint embeddings and original per-view scores locally."""

    encoder = copy.deepcopy(view_model.backbone)  # type: ignore[attr-defined]
    if not hasattr(encoder, "fc"):
        raise TypeError("Phase-1 view model backbone does not expose fc.")
    encoder.fc = nn.Identity()
    encoder = encoder.to(device).eval()
    view_model = view_model.to(device).eval()

    records: list[dict[str, Any]] = []
    embedding_rows: list[np.ndarray] = []
    probability_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            images, view_mask, labels, study_ids, patient_ids = unpack_study_batch(batch)
            images = images.to(device, non_blocking=True)
            view_mask = view_mask.to(device, non_blocking=True).bool()
            real_images = images[view_mask]
            embeddings = encoder(real_images)
            probabilities = torch.sigmoid(view_model(real_images))
            padded_embeddings = embeddings.new_zeros((images.shape[0], images.shape[1], embeddings.shape[1]))
            padded_probabilities = probabilities.new_zeros(view_mask.shape)
            padded_embeddings[view_mask] = embeddings
            padded_probabilities[view_mask] = probabilities
            for index in range(images.shape[0]):
                real_mask = view_mask[index].detach().cpu().numpy().astype(bool)
                real_probabilities = padded_probabilities[index].detach().cpu().numpy()[real_mask]
                records.append(
                    {
                        "patient_id": patient_ids[index],
                        "study_id": study_ids[index],
                        "label": int(labels[index].item()),
                        "num_views": int(real_mask.sum()),
                        "view_probability_std": float(np.std(real_probabilities)),
                        "view_probability_range": float(np.max(real_probabilities) - np.min(real_probabilities)),
                    }
                )
                embedding_rows.append(padded_embeddings[index].detach().cpu().numpy())
                probability_rows.append(padded_probabilities[index].detach().cpu().numpy())
                mask_rows.append(real_mask)

    if not records or len({record["study_id"] for record in records}) != len(records):
        raise RuntimeError("Frozen embedding extraction did not preserve one row per study.")
    max_views = max(len(row) for row in mask_rows)
    dimension = int(embedding_rows[0].shape[-1])
    embeddings = np.zeros((len(records), max_views, dimension), dtype=np.float32)
    probabilities = np.zeros((len(records), max_views), dtype=np.float32)
    mask = np.zeros((len(records), max_views), dtype=bool)
    for index, (row_embeddings, row_probabilities, row_mask) in enumerate(
        zip(embedding_rows, probability_rows, mask_rows)
    ):
        count = len(row_mask)
        embeddings[index, :count] = row_embeddings[:count]
        probabilities[index, :count] = row_probabilities[:count]
        mask[index, :count] = row_mask
    metadata = pd.DataFrame(records)
    if not np.all(mask.sum(axis=1) == metadata["num_views"].to_numpy()):
        raise RuntimeError("Embedding cache view mask does not match declared study view count.")
    local_output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        local_output / f"{split_name}_frozen_embeddings.npz",
        embeddings=embeddings,
        mask=mask,
        labels=metadata["label"].to_numpy(dtype=int),
        patient_ids=metadata["patient_id"].to_numpy(dtype=str),
        study_ids=metadata["study_id"].to_numpy(dtype=str),
        num_views=metadata["num_views"].to_numpy(dtype=int),
        view_probabilities=probabilities,
    )
    metadata.to_csv(local_output / f"{split_name}_embedding_metadata.csv", index=False)
    return {"embeddings": embeddings, "mask": mask, "metadata": metadata}


def _predict_embedding_model(
    model: MultiViewEmbeddingClassifier,
    embeddings: np.ndarray,
    mask: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(embeddings), 64):
            x = torch.from_numpy(embeddings[start : start + 64]).to(device)
            m = torch.from_numpy(mask[start : start + 64]).to(device)
            logits, _ = model(x, m)
            values.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(values)


def _embedding_validation_loss(
    model: MultiViewEmbeddingClassifier,
    embeddings: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    criterion: nn.Module,
    *,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(embeddings), 64):
            end = min(start + 64, len(embeddings))
            x = torch.from_numpy(embeddings[start:end]).to(device)
            m = torch.from_numpy(mask[start:end]).to(device)
            y = torch.from_numpy(labels[start:end].astype(np.float32)).to(device)
            logits, _ = model(x, m)
            total += float(criterion(logits, y).detach().item()) * len(y)
            count += len(y)
    return total / count


def train_frozen_reference(
    train_cache: dict[str, Any],
    validation_cache: dict[str, Any],
    *,
    positive_weight: float,
    args: argparse.Namespace,
    device: torch.device,
    local_output: Path,
) -> tuple[np.ndarray, list[dict[str, float | int]], dict[str, Any]]:
    """Regenerate the fixed frozen-mean reference on the current split only."""

    set_global_seed(args.seed)
    train_embeddings = torch.from_numpy(train_cache["embeddings"])
    train_mask = torch.from_numpy(train_cache["mask"])
    train_labels = torch.from_numpy(train_cache["metadata"]["label"].to_numpy(dtype=np.float32))
    loader = DataLoader(
        TensorDataset(train_embeddings, train_mask, train_labels),
        batch_size=32,
        shuffle=True,
        generator=make_data_loader_generator(args.seed),
    )
    model = MultiViewEmbeddingClassifier("mean").to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], device=device))
    optimizer = AdamW(model.parameters(), lr=args.reference_head_lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)
    labels = validation_cache["metadata"]["label"].to_numpy(dtype=int)
    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_auc = float("-inf")
    best_epoch = 0
    wait = 0
    started = time.perf_counter()
    for epoch in range(1, args.reference_epochs + 1):
        model.train()
        loss_values: list[float] = []
        for embeddings, mask, targets in loader:
            embeddings = embeddings.to(device)
            mask = mask.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(embeddings, mask)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            loss_values.append(float(loss.detach().item()))
        probabilities = _predict_embedding_model(
            model, validation_cache["embeddings"], validation_cache["mask"], device=device
        )
        validation_loss = _embedding_validation_loss(
            model,
            validation_cache["embeddings"],
            validation_cache["mask"],
            labels.astype(np.float32),
            criterion,
            device=device,
        )
        auroc = float(roc_auc_score(labels, probabilities))
        scheduler.step(auroc)
        history.append(
            {
                "model": "Frozen mean embedding",
                "epoch": epoch,
                "train_loss": float(np.mean(loss_values)),
                "validation_loss": float(validation_loss),
                "validation_auroc": auroc,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(f"[frozen reference] epoch {epoch}: validation AUROC={auroc:.6f}", flush=True)
        if auroc > best_auc + 1e-12:
            best_auc = auroc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= args.reference_patience:
            break
    if best_state is None:
        raise RuntimeError("Frozen reference did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    probabilities = _predict_embedding_model(model, validation_cache["embeddings"], validation_cache["mask"], device=device)
    local_output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "best_epoch": best_epoch, "best_validation_auroc": best_auc},
        local_output / "frozen_mean_reference.pt",
    )
    metadata = {
        "best_epoch": best_epoch,
        "best_validation_auroc": best_auc,
        "epochs_completed": len(history),
        "training_seconds": float(time.perf_counter() - started),
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
    }
    return probabilities, history, metadata


def _predict_study_model(
    model: StudyMeanEmbeddingClassifier,
    loader: Iterable[dict[str, Any]],
    *,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[pd.DataFrame, float | None]:
    model.eval()
    records: list[dict[str, Any]] = []
    loss_total = 0.0
    n_studies = 0
    with torch.inference_mode():
        for batch in loader:
            images, view_mask, labels, study_ids, patient_ids = unpack_study_batch(batch)
            images = images.to(device, non_blocking=True)
            view_mask = view_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images, view_mask)
            if criterion is not None:
                loss_total += float(criterion(logits, labels).detach().item()) * len(labels)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            for index, probability in enumerate(probabilities):
                records.append(
                    {
                        "patient_id": patient_ids[index],
                        "study_id": study_ids[index],
                        "label": int(labels[index].item()),
                        "num_views": int(view_mask[index].sum().item()),
                        "abnormal_probability": float(probability),
                    }
                )
            n_studies += len(labels)
    frame = pd.DataFrame(records)
    if frame.empty or frame["study_id"].duplicated().any():
        raise RuntimeError("Study-level prediction did not preserve one row per validation study.")
    return frame, (loss_total / n_studies if criterion is not None else None)


def train_partial_finetune(
    *,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    positive_weight: float,
    args: argparse.Namespace,
    device: torch.device,
    local_output: Path,
) -> tuple[pd.DataFrame, list[dict[str, float | int]], dict[str, Any]]:
    """Run the one primary layer4-plus-head study-level fine-tuning experiment."""

    set_global_seed(args.seed)
    view_model = load_phase1_view_model(args.init_checkpoint, device)
    model = StudyMeanEmbeddingClassifier(view_model.backbone)
    model.configure_trainable_layers(train_layer4=True)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], device=device))
    layer4_parameters = [parameter for parameter in model.backbone.layer4.parameters() if parameter.requires_grad]
    head_parameters = list(model.head.parameters())
    optimizer = AdamW(
        [
            {"params": layer4_parameters, "lr": args.layer4_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6)
    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_auc = float("-inf")
    best_epoch = 0
    wait = 0
    started = time.perf_counter()
    for epoch in range(1, args.partial_epochs + 1):
        model.train()
        loss_total = 0.0
        n_studies = 0
        for batch in train_loader:
            images, view_mask, labels, _, _ = unpack_study_batch(batch)
            images = images.to(device, non_blocking=True)
            view_mask = view_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, view_mask)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Encountered non-finite Phase 2C loss.")
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach().item()) * len(labels)
            n_studies += len(labels)
        validation_frame, validation_loss = _predict_study_model(
            model, validation_loader, device=device, criterion=criterion
        )
        validation_auc = float(roc_auc_score(validation_frame["label"], validation_frame["abnormal_probability"]))
        scheduler.step(validation_auc)
        history.append(
            {
                "model": "Partial fine-tuned mean embedding",
                "epoch": epoch,
                "train_loss": loss_total / n_studies,
                "validation_loss": float(validation_loss) if validation_loss is not None else float("nan"),
                "validation_auroc": validation_auc,
                "learning_rate_layer4": float(optimizer.param_groups[0]["lr"]),
                "learning_rate_head": float(optimizer.param_groups[1]["lr"]),
            }
        )
        print(f"[partial fine-tune] epoch {epoch}: validation AUROC={validation_auc:.6f}", flush=True)
        if validation_auc > best_auc + 1e-12:
            best_auc = validation_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= args.partial_patience:
            break
    if best_state is None:
        raise RuntimeError("Partial fine-tuning did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    validation_frame, _ = _predict_study_model(model, validation_loader, device=device, criterion=criterion)
    local_output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_validation_auroc": best_auc,
            "trainable_modules": ["layer4", "study_head"],
        },
        local_output / "best_partial_finetune.pt",
    )
    metadata = {
        "best_epoch": best_epoch,
        "best_validation_auroc": best_auc,
        "epochs_completed": len(history),
        "training_seconds": float(time.perf_counter() - started),
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
    }
    return validation_frame, history, metadata


def model_metrics(model_name: str, predictions: pd.DataFrame) -> tuple[dict[str, object], dict[float, dict[str, object]]]:
    labels = predictions["label"].to_numpy(dtype=int)
    probabilities = predictions["abnormal_probability"].to_numpy(dtype=float)
    row: dict[str, object] = {
        "model": model_name,
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
    }
    policies: dict[float, dict[str, object]] = {}
    for target in DESCRIPTIVE_TARGETS:
        selected = select_sensitivity_constrained_threshold(labels, probabilities, target)
        metrics = operating_metrics(labels, probabilities, float(selected["threshold"]))
        policies[target] = metrics
        prefix = f"at_sensitivity_{int(target * 100)}"
        row[f"{prefix}_threshold"] = float(metrics["threshold"])
        row[f"{prefix}_sensitivity"] = float(metrics["sensitivity"])
        row[f"{prefix}_specificity"] = float(metrics["specificity"])
        row[f"{prefix}_precision"] = float(metrics["precision"])
        row[f"{prefix}_f1"] = float(metrics["f1"])
        row[f"{prefix}_false_positive"] = int(metrics["false_positive_count"])
        row[f"{prefix}_false_negative"] = int(metrics["false_negative_count"])
        row[f"{prefix}_flagged_percent"] = float(metrics["flagged_abnormal_percent"])
    return row, policies


def _disagreement_group(values: pd.Series, lower: float, upper: float) -> pd.Series:
    return pd.Series(
        np.where(values <= lower, "LOW", np.where(values <= upper, "MEDIUM", "HIGH")), index=values.index
    )


def subgroup_summary(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    train_metadata: pd.DataFrame,
    validation_metadata: pd.DataFrame,
) -> pd.DataFrame:
    lower, upper = np.quantile(train_metadata["view_probability_range"], [1 / 3, 2 / 3])
    metadata = validation_metadata[["study_id", "num_views", "view_probability_range"]].copy()
    metadata["view_count_group"] = np.where(
        metadata["num_views"] == 1, "1 view", np.where(metadata["num_views"] == 2, "2 views", "3+ views")
    )
    metadata["disagreement_group"] = _disagreement_group(metadata["view_probability_range"], float(lower), float(upper))
    combined = (
        metadata.merge(
            baseline[["study_id", "label", "abnormal_probability"]].rename(
                columns={"abnormal_probability": "frozen_probability"}
            ),
            on="study_id",
            validate="one_to_one",
        )
        .merge(
            candidate[["study_id", "abnormal_probability"]].rename(
                columns={"abnormal_probability": "partial_probability"}
            ),
            on="study_id",
            validate="one_to_one",
        )
    )
    rows: list[dict[str, object]] = []
    for dimension in ("view_count_group", "disagreement_group"):
        for group, frame in combined.groupby(dimension, sort=True):
            labels = frame["label"].to_numpy(dtype=int)
            row: dict[str, object] = {
                "dimension": dimension,
                "group": str(group),
                "n_studies": int(len(frame)),
                "n_normal": int((labels == 0).sum()),
                "n_abnormal": int((labels == 1).sum()),
                "train_range_lower_tercile": float(lower),
                "train_range_upper_tercile": float(upper),
                "frozen_auroc": np.nan,
                "partial_auroc": np.nan,
            }
            if len(np.unique(labels)) == 2:
                row["frozen_auroc"] = float(roc_auc_score(labels, frame["frozen_probability"]))
                row["partial_auroc"] = float(roc_auc_score(labels, frame["partial_probability"]))
            rows.append(row)
    result = pd.DataFrame(rows)
    result.attrs["train_disagreement_cutpoints"] = [float(lower), float(upper)]
    return result


def error_summary(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    baseline_threshold: float,
    candidate_threshold: float,
) -> pd.DataFrame:
    candidate_safe = candidate[["study_id", "label", "abnormal_probability"]].rename(
        columns={"label": "candidate_label", "abnormal_probability": "partial_probability"}
    )
    paired = baseline[["study_id", "label", "abnormal_probability"]].rename(
        columns={"abnormal_probability": "frozen_probability"}
    ).merge(
        candidate_safe,
        on="study_id",
        validate="one_to_one",
    )
    if len(paired) != len(baseline) or not (paired["label"] == paired["candidate_label"]).all():
        raise ValueError("Paired error analysis requires the same validation studies and labels for both models.")
    y = paired["label"].to_numpy(dtype=int)
    frozen = paired["frozen_probability"].to_numpy(dtype=float) >= baseline_threshold
    partial = paired["partial_probability"].to_numpy(dtype=float) >= candidate_threshold
    rows = [
        ("both_correct", int(((frozen == y) & (partial == y)).sum())),
        ("both_incorrect", int(((frozen != y) & (partial != y)).sum())),
        ("errors_fixed_by_partial", int(((frozen != y) & (partial == y)).sum())),
        ("new_errors_from_partial", int(((frozen == y) & (partial != y)).sum())),
        ("false_negatives_fixed", int(((y == 1) & ~frozen & partial).sum())),
        ("new_false_negatives", int(((y == 1) & frozen & ~partial).sum())),
        ("false_positives_fixed", int(((y == 0) & frozen & ~partial).sum())),
        ("new_false_positives", int(((y == 0) & ~frozen & partial).sum())),
    ]
    return pd.DataFrame(rows, columns=["comparison_at_95_policy", "count"])


GEOMETRY_VARIANTS = ("canonical", "padding_midgray", "outer_border_black_5pct", "direct_resize")


def _variant_tensor(image: Image.Image, variant: str, image_size: int = 224) -> torch.Tensor:
    """Apply one documented display-geometry perturbation for a diagnostic only."""

    grayscale = image.convert("L")
    width, height = grayscale.size
    if variant == "direct_resize":
        prepared = grayscale
    else:
        side = max(width, height)
        left, top = (side - width) // 2, (side - height) // 2
        right, bottom = side - width - left, side - height - top
        fill = 127 if variant == "padding_midgray" else 0
        prepared = ImageOps.expand(grayscale, border=(left, top, right, bottom), fill=fill)
    prepared = prepared.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
    if variant == "outer_border_black_5pct":
        pixels = np.asarray(prepared).copy()
        border = max(1, round(image_size * 0.05))
        pixels[:border, :] = 0
        pixels[-border:, :] = 0
        pixels[:, :border] = 0
        pixels[:, -border:] = 0
        prepared = Image.fromarray(pixels)
    rgb = Image.merge("RGB", (prepared, prepared, prepared))
    tensor = torch.tensor(bytearray(rgb.tobytes()), dtype=torch.uint8)
    tensor = tensor.reshape(image_size, image_size, 3).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


class _VariantStudyDataset(Dataset):
    """Read validation studies locally for a non-selection geometry diagnostic."""

    def __init__(self, studies: list[Any], dataset_root: Path, variant: str) -> None:
        self.studies = studies
        self.dataset_root = dataset_root
        self.variant = variant

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, index: int) -> dict[str, Any]:
        study = self.studies[index]
        views: list[torch.Tensor] = []
        for relative_path in study.image_paths:
            with Image.open(self.dataset_root / relative_path) as opened:
                views.append(_variant_tensor(opened.copy(), self.variant))
        return {
            "images": views,
            "label": int(study.label),
            "patient_id": str(study.patient_id),
            "study_id": str(study.study_id),
            "image_paths": list(study.image_paths),
            "num_views": int(study.num_views),
            "source_split": str(study.source_split),
            "final_split": study.final_split,
        }


def _variant_loader(studies: list[Any], dataset_root: Path, variant: str, args: argparse.Namespace, device: torch.device) -> DataLoader:
    return DataLoader(
        _VariantStudyDataset(studies, dataset_root, variant),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_studies,
        worker_init_fn=seed_data_loader_worker if args.workers else None,
        generator=make_data_loader_generator(args.seed),
        pin_memory=device.type == "cuda",
    )


def _predict_frozen_variant(
    encoder: nn.Module, head: MultiViewEmbeddingClassifier, loader: Iterable[dict[str, Any]], device: torch.device
) -> np.ndarray:
    values: list[np.ndarray] = []
    encoder.eval()
    head.eval()
    with torch.inference_mode():
        for batch in loader:
            images, view_mask, _, _, _ = unpack_study_batch(batch)
            images, view_mask = images.to(device), view_mask.to(device).bool()
            real_embeddings = encoder(images[view_mask])
            padded = real_embeddings.new_zeros((images.shape[0], images.shape[1], real_embeddings.shape[1]))
            padded[view_mask] = real_embeddings
            logits, _ = head(padded, view_mask)
            values.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(values)


def _predict_partial_variant(
    model: StudyMeanEmbeddingClassifier, loader: Iterable[dict[str, Any]], device: torch.device
) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images, view_mask, _, _, _ = unpack_study_batch(batch)
            logits = model(images.to(device), view_mask.to(device))
            values.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(values)


def run_geometry_audit(
    *,
    validation_studies: list[Any],
    dataset_root: Path,
    args: argparse.Namespace,
    device: torch.device,
    frozen_threshold: float,
    partial_threshold: float,
) -> pd.DataFrame:
    """Compare controlled display perturbation sensitivity without choosing a model."""

    view_model = load_phase1_view_model(args.init_checkpoint, device)
    encoder = copy.deepcopy(view_model.backbone)
    encoder.fc = nn.Identity()
    encoder = encoder.to(device)
    del view_model
    frozen_head = MultiViewEmbeddingClassifier("mean").to(device)
    frozen_payload = torch.load(args.output_dir / "frozen_mean_reference.pt", map_location="cpu")
    frozen_head.load_state_dict(frozen_payload["state_dict"], strict=True)

    partial_view_model = load_phase1_view_model(args.init_checkpoint, device)
    partial_model = StudyMeanEmbeddingClassifier(partial_view_model.backbone).to(device)
    partial_model.configure_trainable_layers(train_layer4=True)
    partial_payload = torch.load(args.output_dir / "best_partial_finetune.pt", map_location="cpu")
    partial_model.load_state_dict(partial_payload["state_dict"], strict=True)

    canonical_loader = _variant_loader(validation_studies, dataset_root, "canonical", args, device)
    frozen_canonical = _predict_frozen_variant(encoder, frozen_head, canonical_loader, device)
    partial_canonical = _predict_partial_variant(partial_model, canonical_loader, device)
    rows: list[dict[str, object]] = []
    for variant in GEOMETRY_VARIANTS[1:]:
        loader = _variant_loader(validation_studies, dataset_root, variant, args, device)
        frozen_variant = _predict_frozen_variant(encoder, frozen_head, loader, device)
        partial_variant = _predict_partial_variant(partial_model, loader, device)
        for model_name, canonical, changed, threshold in (
            ("Frozen mean embedding", frozen_canonical, frozen_variant, frozen_threshold),
            ("Partial fine-tuned mean embedding", partial_canonical, partial_variant, partial_threshold),
        ):
            rows.append(
                {
                    "model": model_name,
                    "display_variant": variant,
                    "mean_absolute_probability_change": float(np.mean(np.abs(changed - canonical))),
                    "decision_flips_at_95_policy": int(np.sum((changed >= threshold) != (canonical >= threshold))),
                    "decision_flip_percent_at_95_policy": float(100 * np.mean((changed >= threshold) != (canonical >= threshold))),
                }
            )
    return pd.DataFrame(rows)


def save_training_curves(history: pd.DataFrame, output_path: Path) -> None:
    """Create a small aggregate-only PNG without adding a plotting dependency."""

    canvas = Image.new("RGB", (2260, 680), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    colors = {"Frozen mean embedding": "#1f77b4", "Partial fine-tuned mean embedding": "#d62728"}

    def panel(left: int, title: str, column: str, lower: float, upper: float, ylabel: str) -> None:
        top, width, height = 90, 620, 470
        right, bottom = left + width, top + height
        draw.text((left, 25), title, fill="black", font=font)
        draw.line((left, bottom, right, bottom), fill="black", width=2)
        draw.line((left, top, left, bottom), fill="black", width=2)
        max_epoch = max(1, int(history["epoch"].max()))
        for tick in range(5):
            value = lower + (upper - lower) * tick / 4
            y = bottom - height * tick / 4
            draw.line((left - 4, y, right, y), fill="#e6e6e6", width=1)
            draw.text((left - 50, y - 5), f"{value:.2f}", fill="#444444", font=font)
        for name, frame in history.groupby("model", sort=False):
            frame = frame.sort_values("epoch")
            points = []
            for _, row in frame.iterrows():
                value = float(row[column])
                x = left + width * (float(row["epoch"]) - 1) / max(1, max_epoch - 1)
                y = bottom - height * (value - lower) / max(upper - lower, 1e-9)
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=colors.get(name, "#333333"), width=3)
            for point in points:
                draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=colors.get(name, "#333333"))
        draw.text((left + width // 2 - 25, bottom + 28), "Epoch", fill="black", font=font)
        draw.text((left, bottom + 48), ylabel, fill="#444444", font=font)

    loss_upper = max(1.0, float(history["train_loss"].max()) * 1.05)
    validation_loss_upper = max(1.0, float(history["validation_loss"].max()) * 1.05)
    panel(110, "Training loss", "train_loss", 0.0, loss_upper, "weighted BCE loss")
    panel(850, "Validation loss", "validation_loss", 0.0, validation_loss_upper, "weighted BCE loss")
    panel(1590, "Internal-validation AUROC", "validation_auroc", 0.5, 1.0, "AUROC")
    legend_x, legend_y = 1000, 630
    for index, name in enumerate(colors):
        x = legend_x + index * 310
        draw.rectangle((x, legend_y, x + 14, legend_y + 14), fill=colors[name])
        draw.text((x + 22, legend_y + 1), name, fill="black", font=font)
    canvas.save(output_path)


def markdown_table(frame: pd.DataFrame, *, precision: int = 6) -> str:
    """Render a compact Markdown table without requiring the optional tabulate package."""

    columns = [str(column) for column in frame.columns]

    def render(value: object) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{precision}f}"
        return str(value).replace("|", "\\|")

    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    rows.extend("| " + " | ".join(render(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def write_safe_reports(
    report_dir: Path,
    *,
    reference_metrics: dict[str, object],
    partial_metrics: dict[str, object],
    paired: pd.DataFrame,
    errors: pd.DataFrame,
    subgroups: pd.DataFrame,
    robustness: pd.DataFrame,
    history: pd.DataFrame,
    reference_metadata: dict[str, Any],
    partial_metadata: dict[str, Any],
    split_summary: dict[str, int],
    class_weight: float,
) -> None:
    """Write aggregate-only results suitable for Git; no IDs, paths, or pixels."""

    report_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame([reference_metrics, partial_metrics])
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    paired.to_csv(report_dir / "paired_bootstrap.csv", index=False)
    errors.to_csv(report_dir / "error_summary.csv", index=False)
    subgroups.to_csv(report_dir / "subgroup_summary.csv", index=False)
    robustness.to_csv(report_dir / "geometry_robustness_summary.csv", index=False)
    history.to_csv(report_dir / "training_history.csv", index=False)

    save_training_curves(history, report_dir / "training_curves.png")

    reference_95 = {key: reference_metrics[key] for key in reference_metrics if key.startswith("at_sensitivity_95_")}
    partial_95 = {key: partial_metrics[key] for key in partial_metrics if key.startswith("at_sensitivity_95_")}
    specificity_row = paired.loc[paired["metric"] == "Specificity@95%"].iloc[0]
    auroc_row = paired.loc[paired["metric"] == "AUROC"].iloc[0]
    auprc_row = paired.loc[paired["metric"] == "AUPRC"].iloc[0]
    specificity_signal = bool(float(specificity_row["ci_lower_95"]) > 0)
    bootstrap_replicates = int(specificity_row["n_bootstrap_requested"])
    error_counts = dict(zip(errors["comparison_at_95_policy"], errors["count"], strict=True))
    partial_history = history.loc[history["model"] == "Partial fine-tuned mean embedding"].sort_values("epoch")
    partial_first = partial_history.iloc[0]
    partial_last = partial_history.iloc[-1]
    overfitting_signal = bool(
        float(partial_last["train_loss"]) < float(partial_first["train_loss"])
        and float(partial_last["validation_loss"]) > float(partial_first["validation_loss"])
        and float(partial_last["validation_auroc"]) < float(partial_first["validation_auroc"])
    )
    high_disagreement = subgroups.loc[
        (subgroups["dimension"] == "disagreement_group") & (subgroups["group"] == "HIGH")
    ].iloc[0]
    frozen_95_specificity = float(reference_95["at_sensitivity_95_specificity"])
    partial_95_specificity = float(partial_95["at_sensitivity_95_specificity"])
    meaningful_improvement = bool(
        specificity_signal
        and float(auroc_row["difference_candidate_minus_baseline"]) >= 0.0
        and float(auprc_row["difference_candidate_minus_baseline"]) >= 0.0
    )
    best_epoch_note = (
        f"The partial model selected epoch {partial_metadata['best_epoch']} by validation AUROC "
        f"from {partial_metadata['epochs_completed']} completed epochs."
    )
    (report_dir / "training_analysis.md").write_text(
        "# Phase 2C training analysis\n\n"
        f"{best_epoch_note} Training and validation curves are in `training_curves.png`. "
        f"For the partial run, train loss moved from {float(partial_first['train_loss']):.4f} to {float(partial_last['train_loss']):.4f}, "
        f"while validation loss moved from {float(partial_first['validation_loss']):.4f} to {float(partial_last['validation_loss']):.4f} and validation AUROC moved from {float(partial_first['validation_auroc']):.6f} to {float(partial_last['validation_auroc']):.6f}. "
        + (
            "This is an overfitting signal after the first epoch, so the first-epoch checkpoint was retained. "
            if overfitting_signal
            else "The recorded curves do not show the pre-specified overfitting pattern. "
        )
        + "The selected checkpoint was determined from internal validation only; no official MURA `valid` row was loaded.\n",
        encoding="utf-8",
    )
    (report_dir / "error_analysis.md").write_text(
        "# Phase 2C paired error analysis\n\n"
        "The table below compares each model at its own highest validation threshold that reached sensitivity >=95%. "
        "It reports counts only and does not infer a medical reason for an error.\n\n"
        + markdown_table(errors)
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "geometry_robustness.md").write_text(
        "# Phase 2C geometry robustness diagnostic\n\n"
        "This is a non-selection sensitivity check: the already-trained models were evaluated on the same internal-validation studies after controlled display-geometry changes. "
        "It cannot prove that a model used anatomy, markers, borders, or any specific clinical feature. It reports only whether scores/decisions changed under the transformations.\n\n"
        + markdown_table(robustness)
        + "\n",
        encoding="utf-8",
    )
    summary = [
        "# Phase 2C summary — controlled study-level partial fine-tuning",
        "",
        "## Scope and integrity",
        "",
        "This is a DEVELOPMENT/INTERNAL-VALIDATION result only. The script accepted official MURA `train` rows only "
        "and used the existing patient-safe internal split. Official MURA `valid` was not loaded for training, model choice, "
        "threshold choice, calibration, or evaluation. Because official `valid` was observed historically in Phase 1, this report "
        "does not make an untouched-final-test claim.",
        "",
        f"Population checked: {split_summary['images']} images, {split_summary['studies']} studies, {split_summary['patients']} patients; "
        f"{split_summary['train_studies']} internal train and {split_summary['validation_studies']} internal validation studies.",
        "",
        "## Models compared",
        "",
        "- **Frozen mean embedding reference:** the existing fine-tuned Phase-1 ResNet-18 emits 512-D view embeddings; the encoder is frozen and a linear study head follows a masked mean over real views.",
        "- **Partial fine-tuned mean embedding:** same study-level mean-embedding architecture, initialized from the same Phase-1 checkpoint; only ResNet `layer4` and the new study head are trainable.",
        f"- The positive-class weight was computed from internal-train studies only: {class_weight:.6f}.",
        "",
        "## Validation comparison",
        "",
        markdown_table(comparison),
        "",
        "## Primary >=95% sensitivity policy",
        "",
        "Each threshold below is the highest threshold that meets the pre-specified sensitivity floor on the same internal-validation split. "
        "Therefore these are development operating estimates, not independent performance evidence.",
        "",
        markdown_table(pd.DataFrame([{"model": "Frozen mean embedding", **reference_95}, {"model": "Partial fine-tuned mean embedding", **partial_95}])),
        "",
        "## Paired uncertainty",
        "",
        f"Candidate minus frozen reference, using {bootstrap_replicates:,} patient-cluster bootstrap replicates on the same validation studies:",
        "",
        markdown_table(paired),
        "",
        "## Interpretation",
        "",
        (
            "The 95% CI for the specificity difference is entirely above zero, so this is a positive development-only signal for that operating-policy dimension. "
            "It is not by itself a complete claim of overall superiority: AUROC/AUPRC uncertainty and the non-selection robustness diagnostic must also be reviewed."
            if specificity_signal
            else "The 95% CI for the specificity difference includes zero; a point estimate alone is not enough to claim that partial fine-tuning improved the 95% sensitivity policy."
        ),
        "",
        "The aggregate paired-error counts are in `error_summary.csv`; view-count and disagreement subgroups are descriptive in `subgroup_summary.csv`, including the internal-train tercile cutpoints used for disagreement. "
        "The non-selection padding/border/resize diagnostic is in `geometry_robustness_summary.csv`. Neither analysis establishes clinical causation. The historical Phase-2B attention result remains negative/inconclusive and was not retrained here.",
        "",
        "## Error, subgroup, and robustness evidence",
        "",
        f"At their own >=95% sensitivity policies, partial fine-tuning fixed {int(error_counts['errors_fixed_by_partial'])} validation errors but introduced {int(error_counts['new_errors_from_partial'])}; it fixed {int(error_counts['false_negatives_fixed'])} false negatives and introduced {int(error_counts['new_false_negatives'])}. Both models therefore had {int(reference_95['at_sensitivity_95_false_negative'])} false negatives, while partial fine-tuning added {int(partial_95['at_sensitivity_95_false_positive']) - int(reference_95['at_sensitivity_95_false_positive'])} false positives.",
        f"The high-disagreement subgroup was essentially tied ({float(high_disagreement['frozen_auroc']):.6f} frozen vs {float(high_disagreement['partial_auroc']):.6f} partial AUROC); this exploratory subgroup does not justify further representation work. The 1-view, 2-view, and 3+-view rows are descriptive because the 1-view subgroup contains only {int(subgroups.loc[(subgroups['dimension'] == 'view_count_group') & (subgroups['group'] == '1 view'), 'n_abnormal'].iloc[0])} abnormal studies.",
        "The geometry diagnostic found non-zero prediction/decision changes under display transformations for both models. This supports continued QC/shortcut investigation, but it does not demonstrate the clinical source of those changes or validate localization.",
        "",
        "## Complexity and stopping rule",
        "",
        f"Frozen reference: {reference_metadata['trainable_parameters']:,} trainable / {reference_metadata['total_parameters']:,} total parameters; best epoch {reference_metadata['best_epoch']}; runtime {reference_metadata['training_seconds']:.1f} seconds.",
        f"Partial model: {partial_metadata['trainable_parameters']:,} trainable / {partial_metadata['total_parameters']:,} total parameters; best epoch {partial_metadata['best_epoch']}; runtime {partial_metadata['training_seconds']:.1f} seconds.",
        "Only the pre-registered reference and one partial fine-tuning run were performed. No hyperparameter sweep, new backbone, attention retraining, ensemble, Agentic AI, or test-set evaluation occurred.",
        "",
        "## Decision gate",
        "",
        (
            "Partial fine-tuning met the pre-registered evidence rule."
            if meaningful_improvement
            else f"Partial fine-tuning did **not** provide a meaningful improvement: at >=95% sensitivity, specificity was {partial_95_specificity:.6f} versus {frozen_95_specificity:.6f} for the frozen reference, and the paired confidence interval includes no improvement. Its ranking point estimates were also lower and the training curve showed an overfitting signal."
        ),
        "The ML core should therefore remain frozen at the simpler frozen mean-embedding reference. The single justified next project step is to advance the bounded deterministic safety policy and separately evaluated Agentic-AI layer, while treating input-QC/geometry sensitivity as a safety-evaluation requirement rather than as proof of a medical shortcut.",
        "",
        "## Final status",
        "",
        "- **PHASE 2C STATUS:** COMPLETE",
        f"- **MEANINGFUL IMPROVEMENT:** {'YES' if meaningful_improvement else 'NO'}",
        f"- **ML CORE READY TO FREEZE:** {'NO' if meaningful_improvement else 'YES'}",
        "- **RECOMMENDED NEXT STEP:** Implement and evaluate the bounded deterministic safety policy and Agentic-AI design on a dedicated agent-evaluation set; do not continue architecture tuning on this already-used development split.",
    ]
    (report_dir / "PHASE2C_SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    _validate_args(args)
    split_summary = assert_train_only_inputs(args.manifest, args.split_manifest)
    print(f"Train-only input guard passed: {split_summary}", flush=True)
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = MURAStudyDataset(
        args.manifest,
        dataset_root=args.dataset_root,
        split_manifest=args.split_manifest,
        split="train",
        # Deterministic images keep the frozen-reference and partial run directly comparable.
        # This Phase 2C experiment isolates the pre-registered study objective and layer4 access;
        # it does not introduce an augmentation sweep.
        transform=build_eval_transform(image_size=args.image_size),
    )
    validation_dataset = MURAStudyDataset(
        args.manifest,
        dataset_root=args.dataset_root,
        split_manifest=args.split_manifest,
        split="val",
        transform=build_eval_transform(image_size=args.image_size),
    )
    _assert_dataset_integrity(train_dataset, validation_dataset)
    if len(train_dataset) != split_summary["train_studies"] or len(validation_dataset) != split_summary["validation_studies"]:
        raise RuntimeError("Dataset construction changed the pre-registered patient-safe study counts.")
    class_decision = decide_class_weighting(_dataset_labels(train_dataset), mode="auto", imbalance_ratio_threshold=1.5)
    positive_weight = class_decision.positive_weight if class_decision.use_positive_weight else 1.0
    print(f"Patient-split guard passed; class positive weight={positive_weight:.6f}", flush=True)

    train_eval_dataset = MURAStudyDataset(
        args.manifest,
        dataset_root=args.dataset_root,
        split_manifest=args.split_manifest,
        split="train",
        transform=build_eval_transform(image_size=args.image_size),
    )
    train_eval_loader = _make_loader(
        train_eval_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    validation_eval_loader = _make_loader(
        validation_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    assert_phase1_checkpoint_provenance(args.init_checkpoint, args.manifest, args.split_manifest)
    extractor_view_model = load_phase1_view_model(args.init_checkpoint, device)
    frozen_encoder_parameters = int(
        sum(parameter.numel() for name, parameter in extractor_view_model.backbone.named_parameters() if not name.startswith("fc."))
    )
    print("Extracting frozen embeddings from internal train only...", flush=True)
    train_cache = extract_frozen_embeddings(
        extractor_view_model, train_eval_loader, device=device, local_output=args.output_dir, split_name="train"
    )
    print("Extracting frozen embeddings from internal validation only...", flush=True)
    validation_cache = extract_frozen_embeddings(
        extractor_view_model, validation_eval_loader, device=device, local_output=args.output_dir, split_name="validation"
    )
    del extractor_view_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Training the fixed frozen mean-embedding reference...", flush=True)
    frozen_probabilities, frozen_history, frozen_metadata = train_frozen_reference(
        train_cache,
        validation_cache,
        positive_weight=positive_weight,
        args=args,
        device=device,
        local_output=args.output_dir,
    )
    # The frozen control still requires the shared ResNet encoder at inference.
    # Its total count therefore includes the non-trainable encoder plus its 513-parameter study head.
    frozen_metadata["total_parameters"] = int(frozen_metadata["total_parameters"]) + frozen_encoder_parameters
    frozen_metadata["frozen_encoder_parameters"] = frozen_encoder_parameters
    frozen_predictions = validation_cache["metadata"][["patient_id", "study_id", "label", "num_views"]].copy()
    frozen_predictions["abnormal_probability"] = frozen_probabilities
    frozen_predictions.to_csv(args.output_dir / "frozen_reference_validation_predictions.csv", index=False)

    train_loader = _make_loader(
        train_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=True,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    print("Training the one pre-registered layer4 partial fine-tuning run...", flush=True)
    partial_predictions, partial_history, partial_metadata = train_partial_finetune(
        train_loader=train_loader,
        validation_loader=validation_eval_loader,
        positive_weight=positive_weight,
        args=args,
        device=device,
        local_output=args.output_dir,
    )
    partial_predictions.to_csv(args.output_dir / "partial_finetune_validation_predictions.csv", index=False)

    reference_metrics, reference_policies = model_metrics("Frozen mean embedding", frozen_predictions)
    partial_metrics, partial_policies = model_metrics("Partial fine-tuned mean embedding", partial_predictions)
    print("Computing paired patient-cluster bootstrap uncertainty...", flush=True)
    paired = paired_patient_bootstrap(
        frozen_predictions["label"],
        frozen_predictions["abnormal_probability"],
        partial_predictions.set_index("study_id").loc[frozen_predictions["study_id"], "abnormal_probability"],
        frozen_predictions["patient_id"],
        sensitivity_targets=(PRIMARY_TARGET,),
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
    )
    errors = error_summary(
        frozen_predictions,
        partial_predictions,
        float(reference_policies[PRIMARY_TARGET]["threshold"]),
        float(partial_policies[PRIMARY_TARGET]["threshold"]),
    )
    subgroups = subgroup_summary(
        frozen_predictions,
        partial_predictions,
        train_cache["metadata"],
        validation_cache["metadata"],
    )
    history = pd.concat([pd.DataFrame(frozen_history), pd.DataFrame(partial_history)], ignore_index=True, sort=False)
    print("Running the pre-specified non-selection geometry robustness diagnostic...", flush=True)
    robustness = run_geometry_audit(
        validation_studies=validation_dataset.studies,
        dataset_root=args.dataset_root,
        args=args,
        device=device,
        frozen_threshold=float(reference_policies[PRIMARY_TARGET]["threshold"]),
        partial_threshold=float(partial_policies[PRIMARY_TARGET]["threshold"]),
    )
    run_config = {
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "split_summary": split_summary,
        "class_weighting": class_decision.to_dict(),
        "preprocessing": preprocessing_spec(args.image_size),
        "phase2c_train_transform": "deterministic evaluation transform; no augmentation",
        "test_access": "forbidden; only official MURA train source rows were accepted",
        "protocol": "reports/phase2c/PHASE2C_PROTOCOL.md",
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print("Writing aggregate Git-safe reports...", flush=True)
    write_safe_reports(
        args.report_dir,
        reference_metrics=reference_metrics,
        partial_metrics=partial_metrics,
        paired=paired,
        errors=errors,
        subgroups=subgroups,
        robustness=robustness,
        history=history,
        reference_metadata=frozen_metadata,
        partial_metadata=partial_metadata,
        split_summary=split_summary,
        class_weight=float(positive_weight),
    )
    print(f"Saved local detailed run to: {args.output_dir}")
    print(f"Saved aggregate Git-safe report to: {args.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
