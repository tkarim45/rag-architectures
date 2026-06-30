"""Hybrid retriever — fuse dense (embedding cosine) and sparse (BM25) rankings with Reciprocal Rank
Fusion. Each arm pulls a deeper 2k candidate pool so the fusion has room to re-rank."""
from __future__ import annotations

from common import providers
from common.retrieval import rrf


def retrieve(query: str, index, k: int = 8) -> list[str]:
    qvec = providers.embed([query])[0]
    dense = [cid for cid, _ in index.dense(qvec, 2 * k)]
    sparse = [cid for cid, _ in index.sparse(query, 2 * k)]
    return rrf([dense, sparse])[:k]
