"""Fan-out retrieval and round-robin merge for multi-query RAG.

Parallelism note: the ``ThreadPoolExecutor`` here is **I/O parallelism**. In production each
per-variant dense search is a network round-trip to a remote vector store and/or embedding
service, and threads overlap those waits well. Against the in-process FAISS/NumPy store it is
merely harmless — the GIL serializes local math — but the code shape matches what a deployed
system needs.

Merge note: per-variant rankings are merged by **round-robin interleave** (rank 1 of every
variant, then rank 2, ...) with first-seen dedup — deliberately NOT by sorting on raw scores.
Scores across variants are comparable-ish (same embedder, same index), but a hard variant's best
hit routinely scores below an easy variant's tenth hit, so a global score-sort lets one phrasing
monopolize the context and defeats the point of asking for variants. Interleaving guarantees
every variant's head makes the pool, preserving per-variant diversity. The rank-aware alternative
— Reciprocal Rank Fusion, which rewards documents that recur *across* variants — is a different
bet on consensus and is implemented as its own architecture (``rag_fusion``).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from core import CorpusIndex, ScoredChunk, Tracer


def retrieve_per_query(index: CorpusIndex, queries: Sequence[str], *, k: int, max_workers: int,
                       tracer: Tracer) -> list[list[ScoredChunk]]:
    """Dense-search every query variant concurrently; rankings come back in query order."""
    with tracer.span("multi_query.fanout", queries=len(queries), k=k) as span:
        workers = max(1, min(max_workers, len(queries)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rankings = list(pool.map(lambda q: index.dense_search(q, k), queries))
        span.set("hits", sum(len(r) for r in rankings))
    return rankings


def interleave(rankings: Sequence[Sequence[ScoredChunk]]) -> list[ScoredChunk]:
    """Round-robin merge with first-seen chunk dedup. Each variant's internal rank order is
    preserved; ties in arrival order break in favor of earlier variants (the original question
    is first, so its hits win position when depths collide)."""
    merged: list[ScoredChunk] = []
    seen: set[str] = set()
    max_depth = max((len(r) for r in rankings), default=0)
    for depth in range(max_depth):
        for ranking in rankings:
            if depth < len(ranking):
                hit = ranking[depth]
                if hit.chunk_id not in seen:
                    seen.add(hit.chunk_id)
                    merged.append(hit)
    return merged
