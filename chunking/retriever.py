"""Chunking retriever — embed the query, return the top-k nearest chunks by cosine. Identical dense
retrieval to naive; the whole point of this architecture is that only the *index* granularity
changes, not the retriever."""
from __future__ import annotations

from common import providers


def retrieve(query: str, index, k: int = 8) -> list[str]:
    qvec = providers.embed([query])[0]
    return [cid for cid, _ in index.dense(qvec, k)]
