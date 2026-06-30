"""Sparse retriever — BM25 keyword scoring over the chunk index (no embeddings)."""
from __future__ import annotations

def retrieve(query: str, index, k: int = 8) -> list[str]:
    return [cid for cid, _ in index.sparse(query, k)]
