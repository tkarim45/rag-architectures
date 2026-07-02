"""Dense retrieval — embed the query, take the top-k nearest chunks, done.

This retriever is deliberately a thin adapter over ``CorpusIndex.dense_search``. It exists as a
named component (rather than an inline call in the pipeline) for two reasons:

* it satisfies the ``core.retrieval.Retriever`` protocol, so the benchmark and other tooling can
  treat "the naive retrieval strategy" as a first-class, swappable object; and
* it is the single place where the baseline's diagnostics story is written — per-hit scores and
  latency — so score distributions can be compared against fancier architectures span-for-span.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from core import CorpusIndex, Query, RetrievalResult, Tracer
from core import tracer as default_tracer


@dataclass
class DenseRetriever:
    """Single-shot dense (embedding cosine) retrieval over a prebuilt :class:`CorpusIndex`.

    Implements ``core.retrieval.Retriever``. Stateless between calls: all offline work (chunking,
    embedding, vector store) already lives in the injected index, which is exactly the
    offline/online split production systems have — this class is the "online" half only.
    """

    index: CorpusIndex
    tracer: Tracer = field(default_factory=lambda: default_tracer)

    def retrieve(self, query: Query) -> RetrievalResult:
        """One embedding call, one ANN lookup — the entire naive retrieval strategy.

        Diagnostics record the ranked (chunk_id, score) pairs and wall-clock latency so the
        benchmark can see *how confidently* dense search ranked its hits, not just which docs
        came back. A flat score distribution here is the tell for the vocabulary-mismatch and
        multi-hop failure modes documented in the package README.
        """
        with self.tracer.span("naive.dense_search", k=query.top_k) as span:
            started = time.perf_counter()
            hits = self.index.dense_search(query.text, query.top_k)
            latency_ms = (time.perf_counter() - started) * 1000.0
            span.set("hits", len(hits))
        return RetrievalResult(
            query=query,
            chunks=hits,
            diagnostics={
                "retriever": "dense",
                "embedding_dim": self.index.embedder.dim,
                "chunker": self.index.strategy,
                "scores": [(h.chunk_id, round(h.score, 4)) for h in hits],
                "latency_ms": round(latency_ms, 2),
            },
        )
