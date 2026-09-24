#!/usr/bin/env python
"""Extract frozen ResNet-18 image features and mean-pool them by MURA study.

Example (uses only the patient-safe train split):

    python scripts/extract_embeddings.py \
      --manifest data/manifests/wrist_manifest.csv \
      --split-manifest data/manifests/split_manifest.csv \
      --split train --output artifacts/embeddings_train.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.collate import collate_images
from src.data.dataset import MURAImageDataset
from src.data.transforms import build_eval_transform
from src.features.extract_embeddings import (
    FrozenResNet18Embedder,
    extract_study_embeddings,
    save_study_embeddings,
)
from src.models.resnet18_classifier import PretrainedWeightsUnavailableError
from src.training.trainer import (
    make_data_loader_generator,
    resolve_device,
    seed_data_loader_worker,
    set_global_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="XR_WRIST image manifest CSV.")
    parser.add_argument(
        "--split-manifest", type=Path, required=True, help="Patient-safe final split manifest CSV."
    )
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output .npz study-embedding artifact.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional root used to resolve relative image paths in the manifest.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a PyTorch device string.")
    parser.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="ImageNet is required for Model A; 'none' is only for a plumbing smoke test.",
    )
    parser.add_argument(
        "--allow-weight-download",
        action="store_true",
        help="Explicitly permit torchvision to download ImageNet weights when the local cache is empty.",
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative.")
    if args.output.suffix.lower() != ".npz":
        raise ValueError("--output must end in .npz.")
    set_global_seed(args.seed)
    dataset = MURAImageDataset(
        args.manifest,
        dataset_root=args.dataset_root,
        split_manifest=args.split_manifest,
        split=args.split,
        transform=build_eval_transform(image_size=args.image_size),
    )
    if len(dataset) == 0:
        raise ValueError(f"The requested {args.split!r} split contains no images.")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_images,
        worker_init_fn=seed_data_loader_worker if args.workers else None,
        generator=make_data_loader_generator(args.seed),
        pin_memory=resolve_device(args.device).type == "cuda",
    )
    embedder = FrozenResNet18Embedder(
        weights=args.weights,
        allow_weight_download=args.allow_weight_download,
    )
    study_embeddings = extract_study_embeddings(loader, embedder, device=resolve_device(args.device))
    study_embeddings.metadata.update(
        {
            "final_split": args.split,
            "manifest": str(args.manifest.resolve()),
            "split_manifest": str(args.split_manifest.resolve()),
            "seed": int(args.seed),
            "image_size": int(args.image_size),
            "image_count": int(len(dataset)),
        }
    )
    output = save_study_embeddings(study_embeddings, args.output)
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(study_embeddings.to_metadata(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = run(args)
    except PretrainedWeightsUnavailableError as exc:
        parser.error(str(exc))
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    print(f"Saved study-level {args.split} embeddings: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
