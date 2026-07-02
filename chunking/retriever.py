"""Dense retrieval over one chunking strategy's index.

The query path here is *deliberately* identical to naive RAG: embed the question, take the top-k
nearest chunks by cosine, nothing else — no fusion, no rerank, no query rewriting. That sameness
is the controlled variable of the whole architecture: when three pipelines share this retriever
and differ only in which index they read, every score delta is attributable to the chunking
strategy alone.

What this retriever adds over naive is evidence: its diagnostics record, for the top hits, the
`index_text` that was matched next to the `display_text` that will be returned — the
match-small/return-big mechanics of each strategy, made visible per query.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from core import (CorpusIndex, Query, RetrievalResult, ScoredChunk, Tracer,
                  tracer as default_tracer)

from .config import Config


def _preview(text: str, limit: int = 160) -> str:
    """Whitespace-normalized prefix for diagnostics — long parent documents would otherwise
    drown the trace output they are meant to explain."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


@dataclass
class DenseStrategyRetriever:
    """Plain dense retriever bound to one strategy's `CorpusIndex`.

    Satisfies the `core.Retriever` protocol. Stateless across queries; safe to share.
    """

    index: CorpusIndex
    config: Config = field(default_factory=Config)
    tracer: Tracer = field(default_factory=lambda: default_tracer)

    def retrieve(self, query: Query) -> RetrievalResult:
        with self.tracer.span("chunking.retrieve.dense", strategy=self.index.strategy,
                              top_k=query.top_k) as span:
            hits = self.index.dense_search(query.text, query.top_k)
            span.set("hits", len(hits))
            span.set("docs", len({h.doc_id for h in hits}))
        return RetrievalResult(query=query, chunks=hits, diagnostics=self._diagnostics(hits))

    # ---- diagnostics ---------------------------------------------------------------------

    def _diagnostics(self, hits: Sequence[ScoredChunk]) -> dict[str, Any]:
        """The strategy's story for this query: which index this was, how granular it is, and —
        for the top hits — the matched index text beside the returned display text."""
        return {
            "strategy": self.index.strategy,
            "n_index_chunks": len(self.index.chunks),
            "n_documents": len(self.index.documents),
            "n_hits": len(hits),
            "matches": [self._match_record(h) for h in hits[: self.config.diagnostics_matches]],
        }

    @staticmethod
    def _match_record(hit: ScoredChunk) -> dict[str, Any]:
        index_text = hit.chunk.index_text
        display_text = hit.chunk.display_text
        return {
            "chunk_id": hit.chunk_id,
            "score": round(hit.score, 4),
            "index_text": _preview(index_text),
            "display_text": _preview(display_text),
            # display/index size ratio: 1.0 = coupled (naive), >1 = match-small/return-big
            "expansion": round(len(display_text) / max(len(index_text), 1), 2),
        }
