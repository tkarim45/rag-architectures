"""Tunables for the naive dense-RAG baseline.

Naive RAG has exactly four knobs, and that is the point of the architecture: everything a more
sophisticated package adds (query rewriting, fusion, reranking, iteration) is an answer to what
these four knobs *cannot* fix. Keeping the config this small makes the baseline honest — there is
no hidden lever to tune it past what single-shot dense retrieval can deliver.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NaiveConfig:
    """Configuration for the naive dense pipeline.

    Attributes:
        chunker: Name of the core chunking strategy used when the pipeline has to build its own
            index (the benchmark normally injects a shared one). ``"sentence"`` is the canonical
            baseline granularity: small enough for precise matches, large enough to be readable.
        top_k: How many chunks dense search returns. Recall rises with k but precision falls —
            every extra passage is one more chance to hand the generator a distractor. 5 is the
            conventional sweet spot for single-hop questions over a small corpus.
        context_max_passages: Upper bound on passages stuffed into the prompt. Kept equal to
            ``top_k`` by default so "what was retrieved" and "what the generator saw" coincide,
            which makes benchmark failures attributable to retrieval rather than truncation.
        context_max_chars: Character budget (~tokens x 4) for the assembled context. A guard
            against pathological chunk sizes, not a tuning lever.
    """

    chunker: str = "sentence"
    top_k: int = 5
    context_max_passages: int = 5
    context_max_chars: int = 6000
