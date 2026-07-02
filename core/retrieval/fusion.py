"""Rank/score fusion — combining multiple retrieval rankings into one.

Two families, both provided because they fail differently:
  * Reciprocal Rank Fusion (RRF) — rank-based, scale-free. The safe default when the input scores
    live on incomparable scales (BM25 vs cosine); only positions matter.
  * Weighted score fusion — normalize each ranking's scores (min-max or z-score), then combine
    with weights. Preserves score *magnitude* information RRF throws away, at the cost of being
    sensitive to normalization choices.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Literal, Sequence

from ..types import ScoredChunk


def rrf(rankings: Sequence[Sequence[ScoredChunk]], *, k: int = 60,
        retriever: str = "rrf") -> list[ScoredChunk]:
    """Reciprocal Rank Fusion (Cormack et al. 2009): score(d) = Σ 1/(k + rank_i(d)).
    `k=60` is the canonical constant; it damps the head so one ranking can't dominate."""
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, ScoredChunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            scores[hit.chunk_id] += 1.0 / (k + rank + 1)
            first_seen.setdefault(hit.chunk_id, hit)
    fused = sorted(scores.items(), key=lambda kv: -kv[1])
    return [ScoredChunk(first_seen[cid].chunk, score, retriever=retriever)
            for cid, score in fused]


def _normalize(values: list[float], method: Literal["minmax", "zscore"]) -> list[float]:
    if not values:
        return values
    if method == "minmax":
        lo, hi = min(values), max(values)
        if hi - lo < 1e-12:
            return [1.0 for _ in values]
        return [(v - lo) / (hi - lo) for v in values]
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = var ** 0.5
    if std < 1e-12:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


def weighted_fusion(rankings: Sequence[Sequence[ScoredChunk]],
                    weights: Sequence[float] | None = None, *,
                    normalization: Literal["minmax", "zscore"] = "minmax",
                    retriever: str = "weighted") -> list[ScoredChunk]:
    """Normalize each ranking's scores onto a shared scale, then take the weighted sum. A document
    absent from a ranking contributes zero from it (standard convex-combination hybrid search)."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"{len(rankings)} rankings but {len(weights)} weights")
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, ScoredChunk] = {}
    for ranking, weight in zip(rankings, weights):
        normalized = _normalize([h.score for h in ranking], normalization)
        for hit, norm_score in zip(ranking, normalized):
            scores[hit.chunk_id] += weight * norm_score
            first_seen.setdefault(hit.chunk_id, hit)
    fused = sorted(scores.items(), key=lambda kv: -kv[1])
    return [ScoredChunk(first_seen[cid].chunk, score, retriever=retriever)
            for cid, score in fused]
