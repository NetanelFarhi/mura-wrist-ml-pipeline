"""Development-only shortcut and geometry audit for MURA XR_WRIST.

This program is deliberately isolated from the repository's combined MURA
manifest.  An audit run must receive explicit internal-development manifests:
either separate train/validation clean manifests, or a train-only manifest with
an explicit train/validation split manifest.  Official MURA ``valid`` data and
any ``test`` data are rejected before images are opened.

The script never trains a model, selects a threshold, or changes a checkpoint.
It measures sensitivity of an already-frozen model to controlled image changes
and reports a geometry-only *warning signal*.  It cannot establish that a
model did or did not use anatomy, nor can it establish lesion localization.
Do not commit the generated CSV files: they contain study identifiers.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VARIANTS = (
    "canonical",
    "padding_midgray",
    "outer_border_black_5pct",
    "direct_resize",
)
ALLOWED_DEVELOPMENT_SPLITS = {"train", "val", "validation"}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-repo-root",
        type=Path,
        help="Repository containing src/models/resnet18_classifier.py. It is used only for model code.",
    )
    parser.add_argument("--checkpoint", type=Path, help="Frozen ResNet-18 checkpoint to audit.")
    parser.add_argument("--data-root", type=Path, help="Local MURA-v1.1 root containing train/XR_WRIST.")
    parser.add_argument("--output-dir", type=Path, help="Local, ignored directory for diagnostic outputs.")

    inputs = parser.add_argument_group("required development-data inputs")
    inputs.add_argument(
        "--train-manifest",
        type=Path,
        help="Clean internal-training image manifest. Use with --validation-manifest.",
    )
    inputs.add_argument(
        "--validation-manifest",
        type=Path,
        help="Clean internal-validation image manifest. Use with --train-manifest.",
    )
    inputs.add_argument(
        "--development-manifest",
        type=Path,
        help="Single train-origin image manifest. Use with --split-manifest.",
    )
    inputs.add_argument(
        "--split-manifest",
        type=Path,
        help="Patient-safe train/validation study split for --development-manifest.",
    )
    parser.add_argument(
        "--saved-validation-predictions",
        type=Path,
        help="Optional internal-validation predictions for a strict checkpoint reproduction check.",
    )
    parser.add_argument(
        "--frozen-threshold",
        type=float,
        help="Optional already-frozen internal-validation threshold. This script never selects one.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--self-test", action="store_true", help="Run synthetic safety checks only.")
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_relative_path(value: object) -> str:
    path = str(value).replace("\\", "/").lstrip("/")
    marker = "MURA-v1.1/"
    return path.split(marker, 1)[1] if marker in path else path


def _require_existing_file(path: Path | None, option: str) -> Path:
    if path is None:
        raise ValueError(f"{option} is required.")
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{option} does not exist: {path}")
    return path


def _binary_label(series: pd.Series, source: Path) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="raise")
        if numeric.isna().any() or not np.isclose(numeric, np.round(numeric)).all():
            raise ValueError(f"Labels in {source} must be integer 0/1 values.")
        labels = numeric.astype(int)
    else:
        mapping = {
            "0": 0,
            "1": 1,
            "negative": 0,
            "normal": 0,
            "positive": 1,
            "abnormal": 1,
        }
        normalized = series.astype(str).str.strip().str.lower()
        if not normalized.isin(mapping).all():
            unexpected = sorted(normalized.loc[~normalized.isin(mapping)].unique().tolist())
            raise ValueError(f"Unsupported labels in {source}: {unexpected}")
        labels = normalized.map(mapping).astype(int)
    if not labels.isin([0, 1]).all():
        raise ValueError(f"Labels in {source} must be binary 0/1.")
    return labels


def _resolve_column(frame: pd.DataFrame, candidates: tuple[str, ...], description: str, source: Path) -> str:
    matches = [column for column in candidates if column in frame.columns]
    if not matches:
        raise ValueError(f"{source} is missing {description}; expected one of {candidates}.")
    return matches[0]


def _assert_train_origin_paths(frame: pd.DataFrame, source: Path) -> None:
    paths = frame["image_path"].map(normalize_relative_path)
    invalid = ~paths.str.startswith("train/XR_WRIST/")
    if invalid.any():
        examples = paths.loc[invalid].head(3).tolist()
        raise ValueError(
            f"{source} is not a train-only XR_WRIST manifest. Rejected image paths: {examples}"
        )
    forbidden = paths.str.lower().str.contains(r"(?:^|/)(?:valid|test)(?:/|$)", regex=True)
    if forbidden.any():
        examples = paths.loc[forbidden].head(3).tolist()
        raise ValueError(f"{source} references official valid/test image paths: {examples}")


def standardize_development_manifest(path: Path, expected_partition: str | None = None) -> pd.DataFrame:
    """Read one explicit clean manifest and reject any non-development rows."""

    source = _require_existing_file(path, "manifest")
    raw = pd.read_csv(source)
    patient_column = _resolve_column(raw, ("patient_id",), "patient_id", source)
    study_column = _resolve_column(raw, ("study_id", "study_uid"), "study_id/study_uid", source)
    image_column = _resolve_column(raw, ("image_path", "file_name", "relative_path", "path"), "image path", source)
    label_column = _resolve_column(raw, ("label_binary", "label"), "binary label", source)
    if raw[[patient_column, study_column, image_column]].isna().any().any():
        raise ValueError(f"{source} has missing patient, study, or image-path values.")
    frame = pd.DataFrame(
        {
            "patient_id": raw[patient_column].astype(str),
            "study_id": raw[study_column].astype(str),
            "image_path": raw[image_column].map(normalize_relative_path),
            "label": _binary_label(raw[label_column], source),
        }
    )
    if frame.isna().any().any() or (frame[["patient_id", "study_id", "image_path"]] == "").any().any():
        raise ValueError(f"{source} has missing patient, study, or image-path values.")
    _assert_train_origin_paths(frame, source)
    for provenance_column in ("source_split", "source_population"):
        if provenance_column in raw.columns:
            values = raw[provenance_column].astype(str).str.strip().str.lower()
            if not values.eq("train").all():
                rejected = sorted(values.loc[~values.eq("train")].unique().tolist())
                raise ValueError(
                    f"{source} contains a non-train {provenance_column}: {rejected}. "
                    "Official valid/test data are not allowed."
                )
    if expected_partition is not None:
        split_column = next((column for column in ("final_split", "split") if column in raw.columns), None)
        if split_column is not None:
            values = raw[split_column].astype(str).str.strip().str.lower()
            allowed = {"train"} if expected_partition == "train" else {"val", "validation"}
            if not values.isin(allowed).all():
                rejected = sorted(values.loc[~values.isin(allowed)].unique().tolist())
                raise ValueError(
                    f"{source} does not match the expected {expected_partition} partition; found {rejected}."
                )
    inconsistent = frame.groupby("study_id")["label"].nunique()
    if (inconsistent > 1).any():
        bad = inconsistent.loc[inconsistent > 1].index[:3].tolist()
        raise ValueError(f"{source} has inconsistent labels within studies: {bad}")
    if frame.duplicated(["patient_id", "study_id", "image_path"]).any():
        raise ValueError(f"{source} contains duplicate image rows.")
    return frame.sort_values(["patient_id", "study_id", "image_path"], kind="stable").reset_index(drop=True)


def _standardize_split_manifest(path: Path) -> pd.DataFrame:
    source = _require_existing_file(path, "--split-manifest")
    raw = pd.read_csv(source)
    patient_column = _resolve_column(raw, ("patient_id",), "patient_id", source)
    study_column = _resolve_column(raw, ("study_id", "study_uid"), "study_id/study_uid", source)
    split_column = _resolve_column(raw, ("final_split", "split"), "final_split/split", source)
    result = pd.DataFrame(
        {
            "patient_id": raw[patient_column].astype(str),
            "study_id": raw[study_column].astype(str),
            "final_split": raw[split_column].astype(str).str.strip().str.lower(),
        }
    )
    if not result["final_split"].isin(ALLOWED_DEVELOPMENT_SPLITS).all():
        rejected = sorted(result.loc[~result["final_split"].isin(ALLOWED_DEVELOPMENT_SPLITS), "final_split"].unique().tolist())
        raise ValueError(
            f"{source} includes disallowed split(s) {rejected}. This audit accepts train/val only, never test."
        )
    if result.duplicated(["patient_id", "study_id"]).any():
        raise ValueError(f"{source} contains duplicate patient/study rows.")
    return result


def _assert_disjoint_development_splits(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    train_patients = set(train["patient_id"])
    validation_patients = set(validation["patient_id"])
    overlap = train_patients & validation_patients
    if overlap:
        raise ValueError(f"Patient overlap detected between internal train/validation: {sorted(overlap)[:3]}")
    train_studies = set(train["study_id"])
    validation_studies = set(validation["study_id"])
    overlap = train_studies & validation_studies
    if overlap:
        raise ValueError(f"Study overlap detected between internal train/validation: {sorted(overlap)[:3]}")
    if train.empty or validation.empty:
        raise ValueError("Both internal train and internal validation must contain at least one image.")


def load_development_frames(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Load one of two explicit, safe development-data contracts.

    Contract A: --train-manifest and --validation-manifest are pre-split clean
    tables. Contract B: --development-manifest and --split-manifest are joined
    explicitly. Neither contract reads repository-wide manifests by default.
    """

    separate = args.train_manifest is not None or args.validation_manifest is not None
    joined = args.development_manifest is not None or args.split_manifest is not None
    if separate and joined:
        raise ValueError("Use either separate clean manifests or development-manifest + split-manifest, not both.")
    if not separate and not joined:
        raise ValueError("Explicit development manifests are required; repository defaults are intentionally unavailable.")
    if separate:
        if args.train_manifest is None or args.validation_manifest is None:
            raise ValueError("--train-manifest and --validation-manifest must be provided together.")
        train = standardize_development_manifest(args.train_manifest, expected_partition="train")
        validation = standardize_development_manifest(args.validation_manifest, expected_partition="validation")
        provenance = {
            "input_contract": "separate_clean_manifests",
            "train_manifest": str(args.train_manifest.expanduser().resolve()),
            "validation_manifest": str(args.validation_manifest.expanduser().resolve()),
        }
    else:
        if args.development_manifest is None or args.split_manifest is None:
            raise ValueError("--development-manifest and --split-manifest must be provided together.")
        development = standardize_development_manifest(args.development_manifest)
        split = _standardize_split_manifest(args.split_manifest)
        merged = development.merge(split, on=["patient_id", "study_id"], how="left", validate="many_to_one")
        if merged["final_split"].isna().any():
            examples = merged.loc[merged["final_split"].isna(), ["patient_id", "study_id"]].drop_duplicates().head(3)
            raise ValueError(f"Split manifest does not cover all development studies: {examples.to_dict('records')}")
        train = merged.loc[merged["final_split"].eq("train"), development.columns].copy()
        validation = merged.loc[merged["final_split"].isin({"val", "validation"}), development.columns].copy()
        provenance = {
            "input_contract": "train_only_manifest_plus_split",
            "development_manifest": str(args.development_manifest.expanduser().resolve()),
            "split_manifest": str(args.split_manifest.expanduser().resolve()),
        }
    _assert_disjoint_development_splits(train, validation)
    return train.reset_index(drop=True), validation.reset_index(drop=True), provenance


def pad_to_square(image: Image.Image, fill: int) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    right = side - width - left
    bottom = side - height - top
    return ImageOps.expand(image, border=(left, top, right, bottom), fill=fill)


def to_model_tensor(image: Image.Image) -> Tensor:
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR)
    rgb = Image.merge("RGB", (resized, resized, resized))
    tensor = torch.tensor(bytearray(rgb.tobytes()), dtype=torch.uint8)
    tensor = tensor.reshape(IMAGE_SIZE, IMAGE_SIZE, 3).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def transform_variant(image: Image.Image, variant: str) -> Tensor:
    grayscale = image.convert("L")
    if variant == "canonical":
        prepared = pad_to_square(grayscale, fill=0)
    elif variant == "padding_midgray":
        prepared = pad_to_square(grayscale, fill=127)
    elif variant == "direct_resize":
        prepared = grayscale
    elif variant == "outer_border_black_5pct":
        prepared = pad_to_square(grayscale, fill=0)
        prepared = prepared.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR)
        pixels = np.asarray(prepared).copy()
        border = max(1, round(IMAGE_SIZE * 0.05))
        pixels[:border, :] = 0
        pixels[-border:, :] = 0
        pixels[:, :border] = 0
        pixels[:, -border:] = 0
        return to_model_tensor(Image.fromarray(pixels))
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return to_model_tensor(prepared)


def image_geometry(image_path: Path) -> dict[str, float]:
    with Image.open(image_path) as image:
        width, height = image.size
    side = max(width, height)
    return {
        "width": int(width),
        "height": int(height),
        "aspect_ratio": float(width / height),
        "padding_fraction": float(1.0 - (width * height) / float(side * side)),
        "is_portrait": int(height > width),
    }


@dataclass(frozen=True)
class StudyRecord:
    patient_id: str
    study_id: str
    label: int
    image_paths: tuple[str, ...]


def make_studies(frame: pd.DataFrame) -> list[StudyRecord]:
    records: list[StudyRecord] = []
    grouped = frame.groupby(["patient_id", "study_id", "label"], sort=True)
    for (patient_id, study_id, label), group in grouped:
        paths = tuple(sorted(group["image_path"].map(normalize_relative_path).tolist()))
        if not paths:
            raise ValueError(f"Study {study_id} has no views.")
        records.append(StudyRecord(str(patient_id), str(study_id), int(label), paths))
    return records


class VariantStudyDataset(Dataset):
    def __init__(self, records: list[StudyRecord], data_root: Path, variant: str):
        self.records = records
        self.data_root = data_root
        self.variant = variant

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        views: list[Tensor] = []
        for relative_path in record.image_paths:
            path = self.data_root / relative_path
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as opened:
                image = opened.copy()
            views.append(transform_variant(image, self.variant))
        return {
            "views": views,
            "patient_id": record.patient_id,
            "study_id": record.study_id,
            "label": record.label,
            "num_views": len(views),
        }


def collate_studies(batch: list[dict[str, object]]) -> dict[str, object]:
    if not batch:
        raise ValueError("Empty batch")
    max_views = max(len(item["views"]) for item in batch)  # type: ignore[arg-type]
    images = torch.zeros((len(batch), max_views, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
    view_mask = torch.zeros((len(batch), max_views), dtype=torch.bool)
    for batch_index, item in enumerate(batch):
        views = item["views"]  # type: ignore[assignment]
        for view_index, tensor in enumerate(views):  # type: ignore[arg-type]
            images[batch_index, view_index] = tensor
            view_mask[batch_index, view_index] = True
    return {
        "images": images,
        "view_mask": view_mask,
        "patient_id": [str(item["patient_id"]) for item in batch],
        "study_id": [str(item["study_id"]) for item in batch],
        "label": torch.tensor([int(item["label"]) for item in batch], dtype=torch.int64),
        "num_views": torch.tensor([int(item["num_views"]) for item in batch], dtype=torch.int64),
    }


def load_model(model_repo_root: Path, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model_repo_root = model_repo_root.expanduser().resolve()
    checkpoint_path = _require_existing_file(checkpoint_path, "--checkpoint")
    if str(model_repo_root) not in sys.path:
        sys.path.insert(0, str(model_repo_root))
    from src.models.resnet18_classifier import build_resnet18_classifier

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint has no model_state_dict: {checkpoint_path}")
    # The checkpoint supplies all learned weights.  Suppress the builder's
    # otherwise misleading "randomly initialized" message for this audit.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Building a randomly initialized ResNet-18.*",
            category=RuntimeWarning,
        )
        model = build_resnet18_classifier(weights="none")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


@torch.inference_mode()
def predict_variant(
    model: torch.nn.Module,
    records: list[StudyRecord],
    data_root: Path,
    variant: str,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> pd.DataFrame:
    loader = DataLoader(
        VariantStudyDataset(records, data_root, variant),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_studies,
        pin_memory=device.type == "cuda",
    )
    rows: list[dict[str, object]] = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)  # type: ignore[union-attr]
        mask = batch["view_mask"].to(device, non_blocking=True)  # type: ignore[union-attr]
        view_probabilities = torch.sigmoid(model(images[mask])).reshape(-1)
        padded_probabilities = torch.zeros(mask.shape, device=device, dtype=view_probabilities.dtype)
        padded_probabilities[mask] = view_probabilities
        study_probabilities = padded_probabilities.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        for patient_id, study_id, label, num_views, probability in zip(
            batch["patient_id"],  # type: ignore[index]
            batch["study_id"],  # type: ignore[index]
            batch["label"].tolist(),  # type: ignore[index]
            batch["num_views"].tolist(),  # type: ignore[index]
            study_probabilities.detach().cpu().tolist(),
        ):
            rows.append(
                {
                    "patient_id": patient_id,
                    "study_id": study_id,
                    "label": int(label),
                    "num_views": int(num_views),
                    f"probability_{variant}": float(probability),
                }
            )
    predictions = pd.DataFrame(rows)
    if not predictions["study_id"].is_unique:
        raise RuntimeError("Audit predictions must contain exactly one row per study.")
    if not predictions[f"probability_{variant}"].between(0.0, 1.0).all():
        raise RuntimeError("Model emitted a non-probability score.")
    return predictions


def build_geometry_table(frame: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        relative_path = normalize_relative_path(row.image_path)
        details = image_geometry(data_root / relative_path)
        rows.append(
            {
                "patient_id": str(row.patient_id),
                "study_id": str(row.study_id),
                "label": int(row.label),
                "image_path": relative_path,
                **details,
            }
        )
    views = pd.DataFrame(rows)
    studies = (
        views.groupby(["patient_id", "study_id", "label"], as_index=False)
        .agg(
            num_views=("image_path", "size"),
            mean_aspect_ratio=("aspect_ratio", "mean"),
            std_aspect_ratio=("aspect_ratio", "std"),
            mean_padding_fraction=("padding_fraction", "mean"),
            max_padding_fraction=("padding_fraction", "max"),
            portrait_fraction=("is_portrait", "mean"),
        )
        .fillna({"std_aspect_ratio": 0.0})
    )
    if not studies["study_id"].is_unique:
        raise RuntimeError("Expected one geometry row per study.")
    return studies


def run_geometry_only_cv(train_geometry: pd.DataFrame, seed: int) -> dict[str, Any]:
    """Fixed, grouped exploratory model; it is not a clinical or selection model."""

    features = [
        "num_views",
        "mean_aspect_ratio",
        "std_aspect_ratio",
        "mean_padding_fraction",
        "max_padding_fraction",
        "portrait_fraction",
    ]
    x = train_geometry[features].to_numpy(dtype=float)
    y = train_geometry["label"].to_numpy(dtype=int)
    groups = train_geometry["patient_id"].astype(str).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError("Geometry audit requires both labels in internal training data.")
    oof = np.full(len(train_geometry), np.nan, dtype=float)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_rows: list[dict[str, Any]] = []
    for fold, (train_index, holdout_index) in enumerate(splitter.split(x, y, groups), start=1):
        estimator = Pipeline(
            [("scale", StandardScaler()), ("model", LogisticRegression(C=1.0, max_iter=2000, random_state=seed))]
        )
        estimator.fit(x[train_index], y[train_index])
        probability = estimator.predict_proba(x[holdout_index])[:, 1]
        oof[holdout_index] = probability
        fold_rows.append(
            {
                "fold": fold,
                "studies": int(len(holdout_index)),
                "patients": int(pd.Series(groups[holdout_index]).nunique()),
                "auroc": float(roc_auc_score(y[holdout_index], probability)),
                "auprc": float(average_precision_score(y[holdout_index], probability)),
            }
        )
    if not np.isfinite(oof).all():
        raise RuntimeError("Geometry CV did not produce one out-of-fold score per train study.")
    return {
        "features": features,
        "fixed_model": "StandardScaler + LogisticRegression(C=1.0)",
        "grouping": "patient_id",
        "seed": int(seed),
        "oof_auroc": float(roc_auc_score(y, oof)),
        "oof_auprc": float(average_precision_score(y, oof)),
        "folds": fold_rows,
    }


def variants_summary(joined: pd.DataFrame, frozen_threshold: float | None) -> pd.DataFrame:
    base = joined["probability_canonical"]
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        column = f"probability_{variant}"
        scores = joined[column]
        delta = scores - base
        row: dict[str, Any] = {
            "variant": variant,
            "diagnostic_study_auroc": float(roc_auc_score(joined["label"], scores)),
            "diagnostic_study_auprc": float(average_precision_score(joined["label"], scores)),
            "mean_absolute_probability_change": float(delta.abs().mean()),
            "median_absolute_probability_change": float(delta.abs().median()),
            "max_absolute_probability_change": float(delta.abs().max()),
            "spearman_rank_correlation_with_canonical": float(base.corr(scores, method="spearman")),
        }
        if frozen_threshold is not None:
            flips = (base >= frozen_threshold) != (scores >= frozen_threshold)
            row["decision_flips_at_frozen_threshold"] = int(flips.sum())
            row["decision_flip_percent_at_frozen_threshold"] = float(100.0 * flips.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def verify_saved_predictions(joined: pd.DataFrame, predictions_path: Path) -> float:
    saved_source = _require_existing_file(predictions_path, "--saved-validation-predictions")
    saved = pd.read_csv(saved_source)
    probability_column = _resolve_column(
        saved, ("abnormal_probability", "study_probability", "probability"), "probability", saved_source
    )
    required = {"patient_id", "study_id"}
    if not required.issubset(saved.columns):
        raise ValueError(f"{saved_source} must include patient_id and study_id.")
    saved = saved[["patient_id", "study_id", probability_column]].copy()
    saved["patient_id"] = saved["patient_id"].astype(str)
    saved["study_id"] = saved["study_id"].astype(str)
    if saved.duplicated(["patient_id", "study_id"]).any():
        raise ValueError(f"{saved_source} has duplicate study predictions.")
    expected_keys = set(zip(joined["patient_id"], joined["study_id"]))
    saved_keys = set(zip(saved["patient_id"], saved["study_id"]))
    if expected_keys != saved_keys:
        raise ValueError(
            "Saved predictions do not match the explicitly supplied internal validation studies; "
            "do not use them as a reproduction reference."
        )
    comparison = joined.merge(saved, on=["patient_id", "study_id"], validate="one_to_one")
    return float((comparison["probability_canonical"] - comparison[probability_column]).abs().max())


def run_self_test() -> None:
    landscape = Image.fromarray(np.full((100, 200), 120, dtype=np.uint8))
    square = pad_to_square(landscape, fill=0)
    gray_square = pad_to_square(landscape, fill=127)
    assert square.size == (200, 200)
    assert np.asarray(square)[0, 100] == 0
    assert np.asarray(gray_square)[0, 100] == 127
    assert np.asarray(square)[100, 100] == 120
    tensor = transform_variant(landscape, "canonical")
    assert tuple(tensor.shape) == (3, IMAGE_SIZE, IMAGE_SIZE)
    batch = collate_studies(
        [
            {"views": [tensor], "patient_id": "p1", "study_id": "s1", "label": 0, "num_views": 1},
            {"views": [tensor, tensor], "patient_id": "p2", "study_id": "s2", "label": 1, "num_views": 2},
        ]
    )
    assert batch["view_mask"].sum(dim=1).tolist() == [1, 2]
    valid = pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "study_id": ["s1", "s2"],
            "image_path": ["train/XR_WRIST/p1/s1/image1.png", "train/XR_WRIST/p2/s2/image1.png"],
            "label": [0, 1],
        }
    )
    _assert_train_origin_paths(valid, Path("synthetic.csv"))
    unsafe = valid.copy()
    unsafe.loc[0, "image_path"] = "valid/XR_WRIST/p1/s1/image1.png"
    try:
        _assert_train_origin_paths(unsafe, Path("unsafe.csv"))
    except ValueError:
        pass
    else:
        raise AssertionError("Official valid path must be rejected.")
    print("Self-test passed: padding, masking, and train-only manifest guards are valid.")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    required = {
        "--model-repo-root": args.model_repo_root,
        "--checkpoint": args.checkpoint,
        "--data-root": args.data_root,
        "--output-dir": args.output_dir,
    }
    missing = [option for option, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing required options: {', '.join(missing)}")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers cannot be negative.")
    if args.frozen_threshold is not None and not 0.0 <= args.frozen_threshold <= 1.0:
        raise ValueError("--frozen-threshold must be between 0 and 1.")

    model_repo_root = args.model_repo_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not (data_root / "train" / "XR_WRIST").is_dir():
        raise FileNotFoundError(f"Expected local MURA train/XR_WRIST below --data-root, got {data_root}")
    try:
        output_dir.relative_to(model_repo_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "--output-dir must be outside the repository because audit files contain "
            "identifiers and per-study predictions. Publish only reviewed aggregate summaries."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)

    print(f"[1/6] Loading explicit internal development manifests on {device}...", flush=True)
    train_frame, validation_frame, input_provenance = load_development_frames(args)
    print("[2/6] Reading internal train/validation image geometry...", flush=True)
    train_geometry = build_geometry_table(train_frame, data_root)
    validation_geometry = build_geometry_table(validation_frame, data_root)
    print("[3/6] Running fixed patient-grouped geometry-only cross-validation...", flush=True)
    geometry_cv = run_geometry_only_cv(train_geometry, args.seed)

    print("[4/6] Loading the frozen fine-tuned ResNet-18 checkpoint...", flush=True)
    model = load_model(model_repo_root, args.checkpoint, device)
    validation_records = make_studies(validation_frame)
    print("[5/6] Running controlled internal-validation perturbations...", flush=True)
    predictions = [
        predict_variant(model, validation_records, data_root, variant, device, args.batch_size, args.workers)
        for variant in VARIANTS
    ]
    joined = predictions[0]
    for frame in predictions[1:]:
        joined = joined.merge(frame, on=["patient_id", "study_id", "label", "num_views"], validate="one_to_one")

    reproduction_error: float | None = None
    if args.saved_validation_predictions is not None:
        reproduction_error = verify_saved_predictions(joined, args.saved_validation_predictions)
        if reproduction_error > 1e-5:
            raise RuntimeError(
                "Canonical audit inference did not reproduce supplied internal-validation predictions. "
                f"Maximum absolute difference: {reproduction_error:.8f}"
            )
    summary = variants_summary(joined, args.frozen_threshold)

    quantiles = train_geometry["mean_padding_fraction"].quantile([1 / 3, 2 / 3]).to_numpy()
    low, high = map(float, quantiles)
    joined = joined.merge(validation_geometry, on=["patient_id", "study_id", "label", "num_views"], validate="one_to_one")
    joined["padding_group"] = pd.cut(
        joined["mean_padding_fraction"],
        bins=[-np.inf, low, high, np.inf],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )
    subgroup_rows: list[dict[str, Any]] = []
    for group, frame in joined.groupby("padding_group", observed=True):
        scores = frame["probability_canonical"]
        row: dict[str, Any] = {
            "padding_group": str(group),
            "studies": int(len(frame)),
            "abnormal_studies": int(frame["label"].sum()),
            "mean_model_probability": float(scores.mean()),
            "mean_abs_change_padding_midgray": float((frame["probability_padding_midgray"] - scores).abs().mean()),
        }
        if args.frozen_threshold is not None:
            row["error_rate_at_frozen_threshold"] = float(
                ((scores >= args.frozen_threshold).astype(int) != frame["label"]).mean()
            )
        subgroup_rows.append(row)
    subgroups = pd.DataFrame(subgroup_rows)

    print("[6/6] Writing local non-image audit artifacts...", flush=True)
    train_geometry.to_csv(output_dir / "train_study_geometry.csv", index=False)
    validation_geometry.to_csv(output_dir / "validation_study_geometry.csv", index=False)
    joined.to_csv(output_dir / "validation_robustness_predictions.csv", index=False)
    summary.to_csv(output_dir / "robustness_summary.csv", index=False)
    subgroups.to_csv(output_dir / "padding_subgroups.csv", index=False)
    report = {
        "scope": "explicit internal train and internal validation only; official MURA valid/test were rejected and not read",
        "input_provenance": input_provenance,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "device": str(device),
        "seed": int(args.seed),
        "validation_studies": int(len(joined)),
        "validation_patients": int(joined["patient_id"].nunique()),
        "frozen_threshold": args.frozen_threshold,
        "saved_prediction_reproduction_checked": args.saved_validation_predictions is not None,
        "canonical_reproduction_max_abs_error": reproduction_error,
        "ranking_metric_status": (
            "Diagnostic context only. The audit does not establish a new independent model-performance result. "
            "Without matching saved internal-validation predictions, do not compare these values with historical model metrics."
            if args.saved_validation_predictions is None
            else "Diagnostic context only; the matching saved internal-validation predictions were used for checkpoint reproduction."
        ),
        "geometry_only_grouped_cv": geometry_cv,
        "perturbations": {
            "padding_midgray": "Changes only square-padding pixels from black to mid-gray.",
            "outer_border_black_5pct": "Masks the outer 5% border after canonical preprocessing; it may also remove anatomy near the edge.",
            "direct_resize": "Removes square padding and changes image geometry; it is a stress test, not an isolated causal intervention.",
        },
        "interpretation_limit": "Diagnostics identify sensitivity to controlled changes. They do not prove anatomy use, shortcut use, clinical utility, or lesion localization.",
    }
    (output_dir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\nRobustness summary:\n", summary.to_string(index=False))
    print("\nPadding subgroups:\n", subgroups.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
