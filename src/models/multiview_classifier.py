"""Small frozen-embedding study classifiers for Phase 2B."""
from __future__ import annotations
from torch import Tensor, nn
from src.models.multiview_pooling import MeanEmbeddingPool, MaxEmbeddingPool
from src.models.gated_attention import GatedAttentionPool

class MultiViewEmbeddingClassifier(nn.Module):
    def __init__(self, pooling:str, embedding_dim:int=512, attention_hidden_dim:int=128):
        super().__init__(); self.pooling_name=pooling
        self.pool = GatedAttentionPool(embedding_dim,attention_hidden_dim) if pooling=='attention' else (MeanEmbeddingPool() if pooling=='mean' else MaxEmbeddingPool())
        self.head=nn.Linear(embedding_dim,1)
    def forward(self, embeddings:Tensor, mask:Tensor)->tuple[Tensor,Tensor|None]:
        output=self.pool(embeddings,mask)
        if self.pooling_name=='attention': representation,weights=output
        else: representation,weights=output,None
        return self.head(representation).squeeze(-1),weights
