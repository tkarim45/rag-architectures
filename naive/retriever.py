"""Naive retriever — embed the query, return the top-k nearest chunks by cosine. The online half of
the pipeline; the offline index it reads is built once in common/index.py."""
from __future__ import annotations

from common import providers


def retrieve(query: str, index, k: int = 8) -> list[str]:
    qvec = providers.embed([query])[0]
    return [cid for cid, _ in index.dense(qvec, k)]
