"""Study-level mean-embedding model used for the bounded Phase 2C experiment.

The module deliberately aggregates *representations*, rather than scalar view
probabilities.  Only real views are encoded; padding never enters the ResNet
or the mean-pooling operation.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn


class StudyMeanEmbeddingClassifier(nn.Module):
    """Shared ResNet encoder, masked mean embedding pool, and one study head."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        if not hasattr(backbone, "fc") or not isinstance(backbone.fc, nn.Linear):
            raise TypeError("Expected a torchvision ResNet-style backbone with a linear fc layer.")
        self.embedding_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(self.embedding_dim, 1)
        self._frozen_module_names: tuple[str, ...] = ()

    @staticmethod
    def _validate_inputs(images: Tensor, view_mask: Tensor) -> tuple[int, int]:
        if images.ndim != 5:
            raise ValueError("images must have shape [batch, views, channels, height, width].")
        batch_size, max_views = int(images.shape[0]), int(images.shape[1])
        if tuple(view_mask.shape) != (batch_size, max_views):
            raise ValueError("view_mask must match the first two image dimensions.")
        if torch.any(view_mask.bool().sum(dim=1) == 0):
            raise ValueError("Every study needs at least one real view.")
        return batch_size, max_views

    def configure_trainable_layers(self, *, train_layer4: bool) -> None:
        """Freeze the encoder or expose only ResNet ``layer4`` for Phase 2C.

        The classifier head always stays trainable.  Frozen BatchNorm modules
        are switched back to evaluation mode during ``train()`` so their
        running statistics cannot drift while images pass through the model.
        """

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self._frozen_module_names = ("conv1", "bn1", "layer1", "layer2", "layer3")
        if train_layer4:
            for parameter in self.backbone.layer4.parameters():
                parameter.requires_grad = True
        for parameter in self.head.parameters():
            parameter.requires_grad = True
        self._keep_frozen_modules_in_eval()

    def _keep_frozen_modules_in_eval(self) -> None:
        for name in self._frozen_module_names:
            module = getattr(self.backbone, name, None)
            if module is not None:
                module.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if mode:
            self._keep_frozen_modules_in_eval()
        return self

    def encode_real_views(self, images: Tensor, view_mask: Tensor) -> Tensor:
        """Encode real views only and restore their padded study layout."""

        batch_size, max_views = self._validate_inputs(images, view_mask)
        mask = view_mask.bool()
        real_images = images[mask]
        real_embeddings = self.backbone(real_images)
        if real_embeddings.ndim != 2 or real_embeddings.shape[1] != self.embedding_dim:
            raise RuntimeError("Backbone did not return one 512-D embedding per real view.")
        embeddings = real_embeddings.new_zeros((batch_size, max_views, self.embedding_dim))
        embeddings[mask] = real_embeddings
        return embeddings

    def forward(self, images: Tensor, view_mask: Tensor) -> Tensor:
        embeddings = self.encode_real_views(images, view_mask)
        mask = view_mask.bool()
        weights = mask.to(dtype=embeddings.dtype).unsqueeze(-1)
        pooled = (embeddings * weights).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1).to(
            embeddings.dtype
        )
        return self.head(pooled).squeeze(-1)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        """Yield only parameters explicitly enabled for optimisation."""

        return (parameter for parameter in self.parameters() if parameter.requires_grad)


__all__ = ["StudyMeanEmbeddingClassifier"]
