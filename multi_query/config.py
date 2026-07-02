"""Tunables for the multi-query RAG architecture.

Frozen so a config can be hashed into a benchmark run manifest and shared across threads without
defensive copying. Every knob the pipeline reads lives here — nothing downstream hardcodes a
number that changes retrieval behavior.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Multi-query RAG tunables.

    The recall/cost trade lives in `n_queries` × `per_query_k`: more variants and deeper
    per-variant search widen the candidate pool (and the bill — every variant is one more dense
    search), while `final_k` caps what the merge hands to the context builder.
    """

    n_queries: int = 3
    """Alternative phrasings requested from the LLM. The original question is always searched
    too, so the fan-out is at most ``n_queries + 1`` dense searches."""

    per_query_k: int = 8
    """Chunks retrieved per query variant, before merging."""

    final_k: int = 8
    """Chunks kept after the round-robin merge — the pool the context builder draws from."""

    dedup_threshold: float = 0.92
    """Embedding-cosine ceiling for keeping a variant. A variant whose embedding is at least this
    similar to an already-kept query would retrieve the same neighborhood; drop it and save the
    search."""

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
