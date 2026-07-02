"""BM25 lexical retrieval over the corpus chunks.

The core :class:`~core.ingestion.index.CorpusIndex` ships with a default-parameter BM25 half, and
when this package's config matches those defaults the retriever simply reuses it — the benchmark's
shared offline artifact. But the moment the config diverges (different k1/b, stemming off, extra
stopwords) the retriever builds its **own** ``BM25Index`` from the injected index's chunks.
Silently answering a tuned config with the default index would make every tuning experiment a
no-op; respecting the config, even at the cost of a one-time rebuild, is the honest behavior.
"""
from __future__ import annotations

import time
from typing import Sequence

from core import Chunk, CorpusIndex, Query, RetrievalResult, ScoredChunk, Tracer
from core import tracer as default_tracer
from core.stores.lexical import Analyzer, BM25Index

from .config import SparseConfig


class BM25Retriever:
    """Okapi BM25 retrieval satisfying the ``core.retrieval.Retriever`` protocol.

    Query-side and index-side analysis are guaranteed identical by construction: both run through
    the same :class:`Analyzer` instance. Analyzer asymmetry (index stems, query doesn't) is the
    classic silent BM25 bug, and this class makes it unrepresentable.
    """

    def __init__(self, index: CorpusIndex, config: SparseConfig | None = None,
                 tracer: Tracer | None = None) -> None:
        self.index = index
        self.config = config or SparseConfig()
        self.tracer = tracer or default_tracer
        self._owns_index = not self.config.matches_core_defaults()
        self._bm25: BM25Index | None = None if self._owns_index else index.bm25
        self._analyzer: Analyzer = self._build_analyzer() if self._owns_index else index.bm25.analyzer

    # ---- offline-ish: config-specific index construction ---------------------------------

    def _build_analyzer(self) -> Analyzer:
        """Extend (never replace) the core stopword list — the default list encodes question
        words ("who", "what") that must stay dead for any English corpus."""
        base = Analyzer()
        return Analyzer(stopwords=base.stopwords | frozenset(w.lower() for w in self.config.extra_stopwords),
                        stem=self.config.stem, min_token_len=self.config.min_token_len)

    def _ensure_bm25(self) -> BM25Index:
        """Lazily build the config-specific BM25 over the *same chunks* as the shared index, so a
        tuned run differs from the default run in parameters only — never in corpus content."""
        if self._bm25 is None:
            chunks: Sequence[Chunk] = self.index.chunks
            with self.tracer.span("sparse.build_bm25", chunks=len(chunks),
                                  k1=self.config.k1, b=self.config.b,
                                  stem=self.config.stem) as span:
                bm25 = BM25Index(analyzer=self._analyzer, k1=self.config.k1, b=self.config.b)
                bm25.add([c.chunk_id for c in chunks], [c.index_text for c in chunks])
                span.set("reason", "config diverges from core BM25 defaults")
            self._bm25 = bm25
        return self._bm25

    # ---- online ---------------------------------------------------------------------------

    def retrieve(self, query: Query) -> RetrievalResult:
        """Score the analyzed query terms against the inverted index; positive scores only.

        Diagnostics tell the lexical story the benchmark needs: the exact terms BM25 actually
        searched (post stopword/stem — an empty list explains a zero-recall query instantly),
        per-hit scores, the k1/b in force, and whether a config-owned index was used.
        """
        bm25 = self._ensure_bm25()
        query_terms = self._analyzer(query.text)
        with self.tracer.span("sparse.bm25_search", k=query.top_k,
                              query_terms=len(query_terms)) as span:
            started = time.perf_counter()
            hits = bm25.search(query.text, query.top_k)
            latency_ms = (time.perf_counter() - started) * 1000.0
            span.set("hits", len(hits))
        scored = [ScoredChunk(self.index.chunk(h.id), h.score, retriever="sparse") for h in hits]
        return RetrievalResult(
            query=query,
            chunks=scored,
            diagnostics={
                "retriever": "bm25",
                "query_terms": query_terms,
                "k1": self.config.k1,
                "b": self.config.b,
                "stem": self.config.stem,
                "custom_bm25_index": self._owns_index,
                "chunker": self.index.strategy,
                "scores": [(h.id, round(h.score, 4)) for h in hits],
                "latency_ms": round(latency_ms, 2),
            },
        )
