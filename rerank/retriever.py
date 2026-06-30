"""Rerank retriever — two-stage retrieve-then-rerank.

Stage 1 (recall, bi-encoder): embed the query and pull a generous candidate set by cosine, cheap and
fast but coarse. Stage 2 (precision, cross-encoder): rescore each `(query, chunk)` pair jointly and
keep the sharpest top-k. The recall set is deliberately wider than what we return so the reranker has
room to promote chunks the bi-encoder ranked low.
"""
from __future__ import annotations

from common import providers

from . import reranker


def retrieve(query: str, index, k: int = 8, recall_k: int = 15,
             model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> list[str]:
    qvec = providers.embed([query])[0]
    n = max(k * 3, recall_k)                                  # generous recall for the reranker
    candidates = index.dense(qvec, n)
    items = [(cid, index.by_cid[cid].index_text) for cid, _ in candidates]
    return reranker.rerank(query, items, model_name)[:k]
