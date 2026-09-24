"""ResNet-18 construction helpers for Phase 1 study-level MURA training.

The project deliberately makes ImageNet initialization an explicit choice.  A
normal ``torchvision`` call downloads weights as a side effect when they are
missing; that is inconvenient for an offline, privacy-conscious dataset
workflow and makes a CPU smoke test unexpectedly depend on the network.
``weights='imagenet'`` therefore requires a cached checkpoint unless
``allow_weight_download=True`` is passed explicitly.  ``weights='none'`` is
available for plumbing/smoke tests only and is not a replacement for the
pretrained Phase 1 experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
import warnings

import torch
from torch import Tensor, nn


WeightMode = Literal["imagenet", "none"]


class PretrainedWeightsUnavailableError(RuntimeError):
    """Raised when pretrained ImageNet weights were requested but are absent."""


@dataclass(frozen=True)
class ResNet18BuildInfo:
    """Serializable provenance for a ResNet-18 initialization."""

    architecture: str
    weights: str
    allow_weight_download: bool
    weight_cache_path: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _torchvision_resnet18_components():
    """Import torchvision lazily so errors explain the actionable dependency."""

    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            "ResNet-18 requires compatible torch and torchvision installations. "
            "Install the project requirements that match the installed PyTorch build."
        ) from exc
    return ResNet18_Weights, resnet18


def normalize_weight_mode(weights: str | WeightMode) -> WeightMode:
    """Normalize the small, intentional set of supported initialization modes."""

    normalized = str(weights).strip().lower().replace("_", "-")
    aliases = {
        "imagenet": "imagenet",
        "imagenet1k": "imagenet",
        "imagenet-1k": "imagenet",
        "pretrained": "imagenet",
        "none": "none",
        "random": "none",
        "random-init": "none",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported ResNet-18 weights mode {weights!r}. "
            "Use 'imagenet' or 'none'."
        )
    return aliases[normalized]  # type: ignore[return-value]


def imagenet_resnet18_cache_path() -> Path:
    """Return the expected torch hub cache location for torchvision ResNet-18."""

    ResNet18_Weights, _ = _torchvision_resnet18_components()
    weight_enum = ResNet18_Weights.IMAGENET1K_V1
    filename = Path(urlparse(weight_enum.url).path).name
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def _resolve_torchvision_weights(
    weights: str | WeightMode,
    *,
    allow_weight_download: bool,
):
    """Resolve a torchvision enum without silently performing a network request."""

    mode = normalize_weight_mode(weights)
    if mode == "none":
        return None, None

    ResNet18_Weights, _ = _torchvision_resnet18_components()
    cache_path = imagenet_resnet18_cache_path()
    if not cache_path.is_file() and not allow_weight_download:
        raise PretrainedWeightsUnavailableError(
            "ImageNet ResNet-18 weights were requested but are not cached at "
            f"{cache_path}. This project does not download external weights by "
            "default. Either place the official torchvision checkpoint in that "
            "cache, rerun with --allow-weight-download after confirming network "
            "access, or use --weights none only for a smoke test."
        )
    return ResNet18_Weights.IMAGENET1K_V1, cache_path


def build_resnet18_backbone(
    *,
    weights: str | WeightMode = "imagenet",
    allow_weight_download: bool = False,
) -> tuple[nn.Module, ResNet18BuildInfo]:
    """Build an unmodified torchvision ResNet-18 backbone.

    The returned model still has the 1,000-way ImageNet classifier.  Callers
    that need embeddings replace ``fc`` with ``nn.Identity``; callers that
    need binary classification replace it with a one-logit linear head.
    """

    mode = normalize_weight_mode(weights)
    weight_enum, cache_path = _resolve_torchvision_weights(
        mode, allow_weight_download=allow_weight_download
    )
    _, resnet18 = _torchvision_resnet18_components()
    model = resnet18(weights=weight_enum)
    if mode == "none":
        warnings.warn(
            "Building a randomly initialized ResNet-18. This is suitable for a "
            "smoke test but not the required pretrained Phase 1 experiment.",
            RuntimeWarning,
            stacklevel=2,
        )
    return model, ResNet18BuildInfo(
        architecture="resnet18",
        weights=mode,
        allow_weight_download=allow_weight_download,
        weight_cache_path=str(cache_path) if cache_path is not None else None,
    )


class ResNet18Classifier(nn.Module):
    """One-logit ResNet-18 view classifier used by the study-level trainer.

    ``forward`` returns logits, not probabilities.  The trainer converts each
    view logit to a probability and aggregates those probabilities per study;
    this avoids treating individual radiographs as independent labels.
    """

    def __init__(self, backbone: nn.Module, build_info: ResNet18BuildInfo):
        super().__init__()
        if not hasattr(backbone, "fc") or not isinstance(backbone.fc, nn.Linear):
            raise TypeError("Expected a torchvision ResNet-18 backbone with a linear fc head.")
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 1)
        self.backbone = backbone
        self.build_info = build_info

    @property
    def embedding_dim(self) -> int:
        """Dimension immediately before the binary classification head."""

        return int(self.backbone.fc.in_features)

    def forward(self, images: Tensor) -> Tensor:
        """Return a scalar logit for each image in ``images``."""

        logits = self.backbone(images)
        return logits.reshape(-1)

    @torch.inference_mode()
    def predict_view_probabilities(self, images: Tensor) -> Tensor:
        """Return sigmoid probabilities for a batch of individual views."""

        was_training = self.training
        self.eval()
        probabilities = torch.sigmoid(self(images))
        self.train(was_training)
        return probabilities

    def provenance(self) -> dict[str, object]:
        """Return checkpoint-friendly construction metadata."""

        return self.build_info.to_dict()


def build_resnet18_classifier(
    *,
    weights: str | WeightMode = "imagenet",
    allow_weight_download: bool = False,
) -> ResNet18Classifier:
    """Create the fine-tunable binary ResNet-18 classifier for Model B."""

    backbone, build_info = build_resnet18_backbone(
        weights=weights, allow_weight_download=allow_weight_download
    )
    return ResNet18Classifier(backbone, build_info)


__all__ = [
    "PretrainedWeightsUnavailableError",
    "ResNet18BuildInfo",
    "ResNet18Classifier",
    "WeightMode",
    "build_resnet18_backbone",
    "build_resnet18_classifier",
    "imagenet_resnet18_cache_path",
    "normalize_weight_mode",
]
