"""Masked gated MIL attention over view embeddings, not scalar probabilities."""
from __future__ import annotations
import torch
from torch import Tensor, nn

class GatedAttentionPool(nn.Module):
    def __init__(self, embedding_dim:int=512, hidden_dim:int=128):
        super().__init__(); self.v=nn.Linear(embedding_dim,hidden_dim); self.u=nn.Linear(embedding_dim,hidden_dim); self.w=nn.Linear(hidden_dim,1,bias=False)
    def forward(self, embeddings:Tensor, mask:Tensor)->tuple[Tensor,Tensor]:
        if embeddings.ndim!=3 or mask.shape!=embeddings.shape[:2] or (mask.sum(1)==0).any(): raise ValueError('Expected nonempty [batch,views,embedding] and matching mask')
        scores=self.w(torch.tanh(self.v(embeddings))*torch.sigmoid(self.u(embeddings))).squeeze(-1)
        scores=scores.masked_fill(~mask,float('-inf')); weights=torch.softmax(scores,dim=1)
        weights=weights*mask.to(weights.dtype) # exactly zero padding, softmax over real views
        return (weights.unsqueeze(-1)*embeddings).sum(1),weights
