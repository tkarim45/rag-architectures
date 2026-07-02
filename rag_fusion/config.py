"""Tunables for the RAG-Fusion architecture.

Frozen so a config can be hashed into a benchmark run manifest and shared across threads without
defensive copying. Every knob the pipeline reads lives here — nothing downstream hardcodes a
number that changes retrieval behavior.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """RAG-Fusion tunables.

    The architecture's signature knob is ``rrf_k``: it damps how much a single ranking's head can
    dominate the fused score, i.e. how strongly fusion demands *consensus across variants* before
    promoting a chunk.
    """

    n_queries: int = 3
    """Broadened search queries requested from the LLM. The original question is always searched
    too, so the fan-out is at most ``n_queries + 1`` dense searches."""

    per_query_k: int = 8
    """Chunks retrieved per query variant, before rank fusion."""

    rrf_k: int = 60
    """Reciprocal Rank Fusion constant: score(d) = Σ 1/(rrf_k + rank). 60 is the canonical value
    from Cormack et al. (2009); smaller sharpens head-of-ranking influence, larger flattens it
    toward pure cross-variant vote counting."""

    final_k: int = 8
    """Chunks kept after fusion — the pool the context builder draws from."""

    dedup_threshold: float = 0.92
    """Embedding-cosine ceiling for keeping a generated query. A query at least this similar to
    an already-kept one contributes a near-identical ranking, which double-counts its votes in
    RRF — worse than useless, so it is dropped."""

    max_workers: int = 4
    """Thread-pool size for the retrieval fan-out (I/O parallelism — see retriever.py)."""

    expansion_max_tokens: int = 300
    """Completion budget for the expansion call; a JSON array of short queries fits easily."""

    chunker: str = "sentence"
    """Which offline chunking strategy to index with when no prebuilt index is injected."""

    max_context_passages: int = 5
    """Passages the context builder packs for the generator."""

    max_context_chars: int = 6000
    """Character budget (≈ tokens × 4) for the assembled context block."""
