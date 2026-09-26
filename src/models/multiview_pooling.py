"""Order-invariant masked embedding pooling for Phase 2B."""
from __future__ import annotations
import torch
from torch import Tensor, nn

class MeanEmbeddingPool(nn.Module):
    def forward(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        weight=mask.to(embeddings.dtype).unsqueeze(-1)
        return (embeddings*weight).sum(1)/mask.sum(1,keepdim=True).clamp_min(1).to(embeddings.dtype)

class MaxEmbeddingPool(nn.Module):
    def forward(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        return embeddings.masked_fill(~mask.unsqueeze(-1),float('-inf')).max(1).values
