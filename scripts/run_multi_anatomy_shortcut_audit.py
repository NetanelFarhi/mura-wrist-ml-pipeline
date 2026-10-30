#!/usr/bin/env python
"""Run a bounded, validation-only peripheral-pixel sensitivity audit.

This diagnostic does not retrain a model, choose an operating threshold, or
claim lesion localization.  It asks a narrower question: after the exact
evaluation preprocessing used during training, do controlled changes to
synthetic padding, the outer border, or the four corners materially change
study-level predictions?  The audit operates only on official-MURA-train
derived internal-validation data and fails closed if canonical predictions do
not reproduce the saved validation predictions for a checkpoint.

Detailed rows containing study identifiers remain in the local output folder.
The aggregate CSV and figure are safe to place in a report because they contain
no image pixels, paths, patient IDs, or study IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.collate import collate_studies
from src.data.multi_anatomy import SUPPORTED_MURA_ANATOMIES
from src.data.study_dataset import MURAStudyDataset
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, build_eval_transform
from src.models.resnet18_classifier import build_resnet18_classifier
from scripts.train_multi_anatomy_finetuned import validate_development_inputs


VARIANTS = (
    "canonical",
    "padding_midgray",
    "outer_border_black_5pct",
    "four_corner_mask_black_10pct",
)
PREDICTION_TOLERANCE = 1e-5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests-dir", type=Path, required=True, help=".../manifests/by_anatomy")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--finetuned-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anatomies", nargs="+", choices=SUPPORTED_MURA_ANATOMIES, default=list(SUPPORTED_MURA_ANATOMIES))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError("PyTorch is required for the shortcut audit.") from exc
    return torch


def resolve_device(requested: str, torch: Any) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_gray(value: float, torch: Any) -> Any:
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (torch.full((3, 1, 1), float(value), dtype=torch.float32) - mean) / std


def apply_tensor_perturbation(tensor: Any, variant: str) -> Any:
    """Apply a post-transform border/corner perturbation without resizing.

    The input is the canonical normalized [3, H, W] evaluation tensor.  The
    black replacement is represented using the same ImageNet normalization as
    the canonical transform.  Padding is handled before resizing by
    :class:`AuditTransform`, because it must affect only synthetic padding.
    """

    torch = _require_torch()
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported audit variant: {variant}")
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("Expected normalized tensor with shape [3, H, W].")
    if variant in {"canonical", "padding_midgray"}:
        return tensor
    result = tensor.clone()
    height, width = result.shape[-2:]
    black = _normalized_gray(0.0, torch).to(dtype=result.dtype, device=result.device)
    if variant == "outer_border_black_5pct":
        border = max(1, int(round(min(height, width) * 0.05)))
        result[:, :border, :] = black
        result[:, -border:, :] = black
        result[:, :, :border] = black
        result[:, :, -border:] = black
        return result
    if variant == "four_corner_mask_black_10pct":
        size = max(1, int(round(min(height, width) * 0.10)))
        result[:, :size, :size] = black
        result[:, :size, -size:] = black
        result[:, -size:, :size] = black
        result[:, -size:, -size:] = black
        return result
    raise AssertionError("Every audit variant must be handled above.")


class AuditTransform:
    """Exact evaluation transform plus one pre-specified diagnostic variant."""

    def __init__(self, variant: str, image_size: int = 224) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"Unsupported audit variant: {variant}")
        self.variant = variant
        self.base = build_eval_transform(image_size=image_size)

    def __call__(self, image: Any) -> Any:
        if self.variant == "canonical":
            return self.base(image)
        if self.variant == "padding_midgray":
            # Reproduce the project transform exactly except for the synthetic
            # square padding value.  A square input has no synthetic padding
            # and is intentionally unchanged.
            from PIL import Image, ImageOps

            grayscale = image.convert("L")
            width, height = grayscale.size
            if width == height:
                return self.base(grayscale)
            size = max(width, height)
            left = (size - width) // 2
            top = (size - height) // 2
            right = size - width - left
            bottom = size - height - top
            padded = ImageOps.expand(grayscale, border=(left, top, right, bottom), fill=127)
            resized = padded.resize((self.base.image_size, self.base.image_size), resample=Image.Resampling.BILINEAR)
            return self.base._to_three_channel_tensor_or_image(resized, Image)
        return apply_tensor_perturbation(self.base(image), self.variant)


def _load_model(checkpoint_path: Path, device: Any) -> Any:
    torch = _require_torch()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain a model state dict: {checkpoint_path}")
    model = build_resnet18_classifier(weights="none")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def predict_studies(
    *,
    manifest_path: Path,
    split_path: Path,
    dataset_root: Path,
    transform: Any,
    model: Any,
    device: Any,
    batch_size: int,
    workers: int,
) -> pd.DataFrame:
    """Return one mean-pooled probability per internal-validation study."""

    torch = _require_torch()
    if batch_size < 1 or workers < 0:
        raise ValueError("batch-size/workers are invalid")
    dataset = MURAStudyDataset(
        manifest_path,
        dataset_root=dataset_root,
        split_manifest=split_path,
        split="val",
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_studies,
        pin_memory=device.type == "cuda",
    )
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=device.type == "cuda")
            view_mask = batch["view_mask"].to(device)
            valid_images = images[view_mask]
            logits = model(valid_images).reshape(-1)
            if logits.numel() != valid_images.shape[0]:
                raise RuntimeError("Model did not emit one logit per real view.")
            view_probabilities = torch.zeros(view_mask.shape, dtype=logits.dtype, device=device)
            view_probabilities[view_mask] = torch.sigmoid(logits)
            denominator = view_mask.sum(dim=1).clamp_min(1)
            study_probabilities = view_probabilities.sum(dim=1) / denominator
            for index, probability in enumerate(study_probabilities.detach().cpu().tolist()):
                rows.append(
                    {
                        "patient_id": str(batch["patient_ids"][index]),
                        "study_id": str(batch["study_ids"][index]),
                        "label": int(batch["labels"][index].item()),
                        "num_views": int(batch["num_views"][index]),
                        "abnormal_probability": float(probability),
                    }
                )
    frame = pd.DataFrame(rows).sort_values("study_id", kind="stable").reset_index(drop=True)
    if frame.empty or frame["study_id"].duplicated().any() or not frame["num_views"].ge(1).all():
        raise RuntimeError("Study-level prediction export failed its identity contract.")
    return frame


def _read_saved_predictions(path: Path, anatomy: str) -> pd.DataFrame:
    frame = pd.read_csv(path, float_precision="round_trip")
    required = {"anatomy", "split", "patient_id", "study_id", "label", "num_views", "abnormal_probability"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Saved validation predictions missing {sorted(missing)}")
    if not frame["anatomy"].eq(anatomy).all() or not frame["split"].eq("internal_validation").all():
        raise ValueError("Saved predictions are not exclusively this anatomy's internal validation rows.")
    return frame.sort_values("study_id", kind="stable").reset_index(drop=True)


def assert_canonical_match(canonical: pd.DataFrame, saved: pd.DataFrame, tolerance: float = PREDICTION_TOLERANCE) -> None:
    """Fail closed unless exact preprocessing/model inference reproduces saved rows."""

    for field in ("patient_id", "study_id", "label", "num_views"):
        if not canonical[field].equals(saved[field]):
            raise RuntimeError(f"Canonical audit prediction does not match saved {field!r} identity.")
    difference = np.abs(
        canonical["abnormal_probability"].to_numpy(dtype=float)
        - saved["abnormal_probability"].to_numpy(dtype=float)
    )
    maximum = float(difference.max(initial=0.0))
    if maximum > tolerance:
        raise RuntimeError(
            f"Canonical predictions differ from saved validation predictions (max |Δ|={maximum:.3g}); "
            "do not report a shortcut audit."
        )


def bootstrap_variant(
    detailed: pd.DataFrame, *, replicates: int, seed: int
) -> pd.DataFrame:
    """Patient-cluster CIs for perturbation magnitude and decision-flip rate."""

    if replicates < 1:
        raise ValueError("bootstrap-replicates must be positive")
    patients = detailed["patient_id"].astype(str).to_numpy()
    unique = np.unique(patients)
    indices_by_patient = {patient: np.flatnonzero(patients == patient) for patient in unique}
    changes = detailed["abs_probability_change"].to_numpy(dtype=float)
    flips = detailed["decision_changed"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples: list[tuple[float, float]] = []
    for _ in range(replicates):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([indices_by_patient[patient] for patient in selected])
        samples.append((float(changes[indices].mean()), float(100.0 * flips[indices].mean())))
    point_change = float(changes.mean())
    point_flip = float(100.0 * flips.mean())
    array = np.asarray(samples, dtype=float)
    return pd.DataFrame(
        [
            {
                "metric": "mean_abs_probability_change",
                "point_estimate": point_change,
                "lower_95_ci": float(np.quantile(array[:, 0], 0.025)),
                "upper_95_ci": float(np.quantile(array[:, 0], 0.975)),
                "valid_resamples": len(array),
                "requested_resamples": replicates,
                "bootstrap_unit": "patient",
            },
            {
                "metric": "decision_flip_percent",
                "point_estimate": point_flip,
                "lower_95_ci": float(np.quantile(array[:, 1], 0.025)),
                "upper_95_ci": float(np.quantile(array[:, 1], 0.975)),
                "valid_resamples": len(array),
                "requested_resamples": replicates,
                "bootstrap_unit": "patient",
            },
        ]
    )


def summarize_variant(
    *,
    anatomy: str,
    variant: str,
    canonical: pd.DataFrame,
    altered: pd.DataFrame,
    threshold: float,
    checkpoint_sha256: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    if not canonical["study_id"].equals(altered["study_id"]):
        raise RuntimeError("Perturbed prediction rows no longer align with canonical study IDs.")
    probability = altered["abnormal_probability"].to_numpy(dtype=float)
    canonical_probability = canonical["abnormal_probability"].to_numpy(dtype=float)
    labels = canonical["label"].to_numpy(dtype=int)
    canonical_predicted = canonical_probability >= threshold
    altered_predicted = probability >= threshold
    detailed = canonical[["patient_id", "study_id", "label", "num_views"]].copy()
    detailed["canonical_probability"] = canonical_probability
    detailed["variant_probability"] = probability
    detailed["abs_probability_change"] = np.abs(probability - canonical_probability)
    detailed["decision_changed"] = canonical_predicted != altered_predicted
    detailed["canonical_predicted_abnormal"] = canonical_predicted
    detailed["variant_predicted_abnormal"] = altered_predicted
    detailed["new_false_negative"] = (labels == 1) & canonical_predicted & ~altered_predicted
    detailed["new_false_positive"] = (labels == 0) & ~canonical_predicted & altered_predicted
    rank_correlation = float(
        pd.Series(canonical_probability).rank(method="average").corr(pd.Series(probability).rank(method="average"))
    )
    summary = {
        "anatomy": anatomy,
        "variant": variant,
        "variant_category": "canonical" if variant == "canonical" else "peripheral_pixel_sensitivity",
        "evaluation_population": "internal_validation",
        "n_studies": int(len(detailed)),
        "n_patients": int(detailed["patient_id"].nunique()),
        "n_views": int(detailed["num_views"].sum()),
        "frozen_threshold": float(threshold),
        "threshold_source": "saved internal-validation selection record",
        "checkpoint_sha256": checkpoint_sha256,
        "mean_abs_probability_change": float(detailed["abs_probability_change"].mean()),
        "median_abs_probability_change": float(detailed["abs_probability_change"].median()),
        "p95_abs_probability_change": float(detailed["abs_probability_change"].quantile(0.95)),
        "max_abs_probability_change": float(detailed["abs_probability_change"].max()),
        "spearman_rank_correlation": rank_correlation,
        "decision_flip_count": int(detailed["decision_changed"].sum()),
        "decision_flip_percent": float(100.0 * detailed["decision_changed"].mean()),
        "normal_to_abnormal_flip_count": int(((labels == 0) & ~canonical_predicted & altered_predicted).sum()),
        "abnormal_to_normal_flip_count": int(((labels == 1) & canonical_predicted & ~altered_predicted).sum()),
        "new_false_negatives_after_perturbation": int(detailed["new_false_negative"].sum()),
        "new_false_positives_after_perturbation": int(detailed["new_false_positive"].sum()),
    }
    return summary, detailed


def save_aggregate_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    shown = summary.loc[summary["variant"] != "canonical"].copy()
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    pivot_change = shown.pivot(index="anatomy", columns="variant", values="mean_abs_probability_change")
    pivot_flip = shown.pivot(index="anatomy", columns="variant", values="decision_flip_percent")
    pivot_change.plot(kind="bar", ax=axes[0], width=0.82)
    axes[0].set(title="Mean absolute probability change", xlabel="Anatomy", ylabel="Mean |Δ probability|")
    axes[0].grid(axis="y", alpha=0.25)
    pivot_flip.plot(kind="bar", ax=axes[1], width=0.82)
    axes[1].set(title="Decision changes at frozen 95% sensitivity policy", xlabel="Anatomy", ylabel="Studies with a changed decision (%)")
    axes[1].grid(axis="y", alpha=0.25)
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
        axis.legend(fontsize=8, title="Perturbation")
    figure.suptitle("Development-only peripheral-pixel sensitivity audit", y=1.02, fontsize=13)
    figure.tight_layout()
    figure.savefig(output_dir / "shortcut_audit_summary.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    display_columns = [
        "anatomy",
        "variant",
        "mean_abs_probability_change",
        "decision_flip_percent",
        "new_false_negatives_after_perturbation",
        "new_false_positives_after_perturbation",
    ]
    # Keep the diagnostic runnable in the minimal GPU environment as well:
    # pandas.DataFrame.to_markdown() would require the optional ``tabulate``
    # package, which is not part of the project runtime.
    display = summary.loc[summary["variant"] != "canonical", display_columns].copy()
    table_lines = ["| " + " | ".join(display_columns) + " |", "| " + " | ".join(["---"] * len(display_columns)) + " |"]
    for row in display.itertuples(index=False, name=None):
        formatted = [f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        table_lines.append("| " + " | ".join(formatted) + " |")
    table = "\n".join(table_lines)
    report = (
        "# Multi-anatomy peripheral-pixel sensitivity audit\n\n"
        "Scope: official-MURA-train-derived internal validation only. No test labels, new threshold, retraining, or model selection was used.\n\n"
        "The audit first reproduces each saved canonical validation prediction. It then applies three pre-specified image perturbations: mid-gray synthetic padding, a 5% black outer border, and black 10% corner masks.\n\n"
        "## Aggregate results\n\n"
        f"{table}\n\n"
        "## Interpretation boundary\n\n"
        "These are sensitivity-to-perturbation diagnostics, not evidence that a model uses a particular L/R marker, detects a lesion, or is free of shortcut learning. Border/corner masks can also remove anatomy, collimation, or acquisition information.\n"
    )
    (output_dir / "MULTI_ANATOMY_SHORTCUT_AUDIT.md").write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    if args.batch_size < 1 or args.workers < 0 or args.bootstrap_replicates < 1:
        raise ValueError("batch-size, workers, and bootstrap-replicates must be positive/valid")
    manifests_dir = args.manifests_dir.resolve()
    finetuned_dir = args.finetuned_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    if not manifests_dir.is_dir() or not finetuned_dir.is_dir() or not dataset_root.is_dir():
        raise FileNotFoundError("manifests-dir, finetuned-dir, and dataset-root must exist")
    provenance_path = finetuned_dir / "finetuned_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("official_valid_read") is not False or provenance.get("test_used_for_selection") is not False:
        raise ValueError("Shortcut audit accepts only train-derived development artifacts.")
    torch = _require_torch()
    device = resolve_device(args.device, torch)
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_dir = output_dir / "local_detailed_rows"
    detailed_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    intervals: list[pd.DataFrame] = []
    for anatomy in args.anatomies:
        stem = anatomy.lower()
        manifest_path = manifests_dir / f"{stem}_manifest.csv"
        split_path = manifests_dir / f"{stem}_split.csv"
        validate_development_inputs(manifest_path, split_path, anatomy)
        anatomy_dir = finetuned_dir / stem
        checkpoint_path = anatomy_dir / "best_resnet18_study_level.pt"
        selection_path = anatomy_dir / "selection.json"
        saved_path = anatomy_dir / "validation_predictions.csv"
        for path in (checkpoint_path, selection_path, saved_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("test_used_for_selection") is not False:
            raise ValueError(f"{anatomy} threshold selection does not rule out test use.")
        threshold = float(selection["threshold_selection"]["threshold"])
        checkpoint_hash_before = sha256_file(checkpoint_path)
        model = _load_model(checkpoint_path, device)
        canonical = predict_studies(
            manifest_path=manifest_path,
            split_path=split_path,
            dataset_root=dataset_root,
            transform=AuditTransform("canonical"),
            model=model,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        saved = _read_saved_predictions(saved_path, anatomy)
        assert_canonical_match(canonical, saved)
        summary, detailed = summarize_variant(
            anatomy=anatomy,
            variant="canonical",
            canonical=canonical,
            altered=canonical,
            threshold=threshold,
            checkpoint_sha256=checkpoint_hash_before,
        )
        summaries.append(summary)
        for variant_index, variant in enumerate(VARIANTS[1:], start=1):
            altered = predict_studies(
                manifest_path=manifest_path,
                split_path=split_path,
                dataset_root=dataset_root,
                transform=AuditTransform(variant),
                model=model,
                device=device,
                batch_size=args.batch_size,
                workers=args.workers,
            )
            summary, detailed = summarize_variant(
                anatomy=anatomy,
                variant=variant,
                canonical=canonical,
                altered=altered,
                threshold=threshold,
                checkpoint_sha256=checkpoint_hash_before,
            )
            summaries.append(summary)
            detailed.to_csv(detailed_dir / f"{stem}_{variant}.csv", index=False, float_format="%.17g")
            interval = bootstrap_variant(detailed, replicates=args.bootstrap_replicates, seed=args.seed + variant_index)
            interval.insert(0, "anatomy", anatomy)
            interval.insert(1, "variant", variant)
            intervals.append(interval)
        checkpoint_hash_after = sha256_file(checkpoint_path)
        if checkpoint_hash_after != checkpoint_hash_before:
            raise RuntimeError(f"Checkpoint changed during audit: {checkpoint_path}")
        print(f"[{anatomy}] canonical predictions reproduced; three perturbation variants evaluated")
    summary_frame = pd.DataFrame(summaries).sort_values(["anatomy", "variant"], kind="stable")
    interval_frame = pd.concat(intervals, ignore_index=True).sort_values(["anatomy", "variant", "metric"], kind="stable")
    summary_frame.to_csv(output_dir / "MULTI_ANATOMY_SHORTCUT_AUDIT_SUMMARY.csv", index=False, float_format="%.17g")
    interval_frame.to_csv(output_dir / "shortcut_audit_bootstrap_ci.csv", index=False, float_format="%.17g")
    save_aggregate_figure(summary_frame, output_dir)
    _write_report(summary_frame, output_dir)
    audit_provenance = {
        "scope": "development-only internal validation peripheral-pixel sensitivity audit",
        "official_valid_read": False,
        "test_used_for_selection": False,
        "threshold_reselected": False,
        "variants": list(VARIANTS),
        "canonical_prediction_tolerance": PREDICTION_TOLERANCE,
        "bootstrap_unit": "patient",
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "detailed_rows_local_only": True,
    }
    (output_dir / "shortcut_audit_provenance.json").write_text(json.dumps(audit_provenance, indent=2) + "\n", encoding="utf-8")
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = run(args)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(f"Saved local shortcut audit: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
