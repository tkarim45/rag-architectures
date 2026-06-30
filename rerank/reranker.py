"""Cross-encoder reranker — the precision half of two-stage retrieval.

A bi-encoder (the dense recall step) embeds the query and each chunk *separately*, so it can only
measure coarse vector similarity and never sees the two texts together. A cross-encoder feeds the
`(query, chunk)` pair through one transformer and directly predicts a relevance score — far sharper
ordering at the cost of a second model pass over the candidate set. We run it only over the small
recalled set, then keep the best.

The model is loaded once and cached at module level (lazy singleton): the first `rerank` call pays
the load, every call after reuses it.
"""
from __future__ import annotations

import numpy as np

_MODELS: dict[str, object] = {}


def _model(model_name: str):
    """Lazily load and cache the CrossEncoder for `model_name` (loaded once per process)."""
    if model_name not in _MODELS:
        from sentence_transformers import CrossEncoder      # imported lazily — heavy dependency
        _MODELS[model_name] = CrossEncoder(model_name)
    return _MODELS[model_name]


def rerank(query: str, items: list[tuple[str, str]], model_name: str) -> list[str]:
    """Rescore recalled candidates with a cross-encoder and return cids, best first.

    `items` are `(cid, text)` pairs from the dense recall stage. We score every `(query, text)` pair
    jointly, then sort the cids by descending relevance.
    """
    if not items:
        return []
    cids = [cid for cid, _ in items]
    pairs = [(query, text) for _, text in items]
    scores = np.asarray(_model(model_name).predict(pairs))
    order = np.argsort(scores)[::-1]                          # descending relevance
    return [cids[i] for i in order]
