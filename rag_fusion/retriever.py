"""Fan-out retrieval and Reciprocal Rank Fusion for RAG-Fusion.

Parallelism note: the ``ThreadPoolExecutor`` here is **I/O parallelism**. In production each
per-query dense search is a network round-trip to a remote vector store and/or embedding service,
and threads overlap those waits well. Against the in-process FAISS/NumPy store it is merely
harmless — the GIL serializes local math — but the code shape matches what a deployed system
needs.

Fusion note: per-query rankings are combined with RRF (``core.retrieval.fusion.rrf``):
``score(d) = Σ_q 1/(rrf_k + rank_q(d))``. Being rank-based it is scale-free — raw cosine scores
across differently-broadened queries never need to be comparable — and it is a *consensus* scorer:
a chunk that appears mid-ranking in many query variants outscores a chunk that tops exactly one.
That is the architectural bet that distinguishes RAG-Fusion from multi_query's round-robin
interleave, which instead guarantees every variant's head a slot regardless of agreement.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from core import CorpusIndex, ScoredChunk, Tracer, rrf


def retrieve_per_query(index: CorpusIndex, queries: Sequence[str], *, k: int, max_workers: int,
                       tracer: Tracer) -> list[list[ScoredChunk]]:
    """Dense-search every query concurrently; rankings come back in query order."""
    with tracer.span("rag_fusion.fanout", queries=len(queries), k=k) as span:
        workers = max(1, min(max_workers, len(queries)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rankings = list(pool.map(lambda q: index.dense_search(q, k), queries))
        span.set("hits", sum(len(r) for r in rankings))
    return rankings


def fuse(rankings: Sequence[Sequence[ScoredChunk]], *, rrf_k: int,
         tracer: Tracer) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion across the per-query rankings (Cormack et al. 2009). Dedup is
    inherent: RRF keys on chunk_id, so a chunk found by several queries becomes one entry whose
    score aggregates its votes."""
    with tracer.span("rag_fusion.fuse", rankings=len(rankings), rrf_k=rrf_k) as span:
        fused = rrf(rankings, k=rrf_k, retriever="rag_fusion")
        span.set("fused_pool", len(fused))
    return fused
