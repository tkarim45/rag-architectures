"""HybridRetriever — run dense and BM25 branches over the same chunks, fuse into one ranking.

Design decisions:

* **Both branches, always.** Dense retrieval matches paraphrase and intent but misses exact rare
  tokens (ids, product names, error codes); BM25 nails exact tokens but is blind to synonyms.
  Their failure modes are complementary, so hybrid's win condition is *disagreement* — when both
  branches already return the same chunks, fusion is a no-op and hybrid == naive.
* **Fusion strategy is config, not code.** RRF and weighted fusion share a call shape in
  ``core.retrieval.fusion``; the retriever just dispatches on ``config.fusion``. That keeps the
  RRF-vs-weighted experiment a one-line config change the benchmark can sweep.
* **Diagnostics tell the fusion story.** Each branch's full ranking (ids + scores) plus the
  overlap count is recorded in ``RetrievalResult.diagnostics``, so a bad fused ranking can be
  attributed to a bad branch vs a bad fusion choice without re-running the query.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core import (CorpusIndex, Query, RetrievalResult, ScoredChunk, Tracer, rrf, tracer,
                  weighted_fusion)

from .config import Config


def _branch_summary(hits: list[ScoredChunk]) -> list[dict[str, Any]]:
    """Compact (chunk_id, doc_id, score) view of one branch's ranking for diagnostics."""
    return [{"chunk_id": h.chunk_id, "doc_id": h.doc_id, "score": round(h.score, 4)}
            for h in hits]


@dataclass
class HybridRetriever:
    """Two-branch retrieval + fusion. Stateless between queries; owns no index lifecycle —
    the index is injected so the benchmark can share one across architectures."""

    index: CorpusIndex
    config: Config = field(default_factory=Config)
    tracer: Tracer = field(default_factory=lambda: tracer)

    def retrieve(self, query: Query) -> RetrievalResult:
        cfg = self.config

        with self.tracer.span("hybrid.dense", k=cfg.dense_k) as span:
            dense = self.index.dense_search(query.text, cfg.dense_k)
            span.set("hits", len(dense))

        with self.tracer.span("hybrid.sparse", k=cfg.sparse_k) as span:
            sparse = self.index.sparse_search(query.text, cfg.sparse_k)
            span.set("hits", len(sparse))

        with self.tracer.span("hybrid.fuse", method=cfg.fusion) as span:
            fused = self._fuse(dense, sparse)
            final = fused[:cfg.final_k]
            span.set("fused", len(fused))
            span.set("kept", len(final))

        overlap = {h.chunk_id for h in dense} & {h.chunk_id for h in sparse}
        diagnostics: dict[str, Any] = {
            "fusion": self._fusion_settings(),
            "dense": {"k": cfg.dense_k, "top": _branch_summary(dense)},
            "sparse": {"k": cfg.sparse_k, "top": _branch_summary(sparse)},
            "branch_overlap": len(overlap),          # 0 ⇒ branches fully disagree; fusion decides
            "fused_total": len(fused),
            "final_k": cfg.final_k,
        }
        return RetrievalResult(query=query, chunks=final, diagnostics=diagnostics)

    # ------------------------------------------------------------------------------------

    def _fuse(self, dense: list[ScoredChunk], sparse: list[ScoredChunk]) -> list[ScoredChunk]:
        """Dispatch to the configured fusion. RRF ignores score magnitudes (scale-free);
        weighted fusion normalizes each branch onto [0,1] / z-scores first, then mixes."""
        if self.config.fusion == "rrf":
            return rrf([dense, sparse], k=self.config.rrf_k, retriever="hybrid.rrf")
        return weighted_fusion([dense, sparse], list(self.config.weights),
                               normalization=self.config.normalization,
                               retriever="hybrid.weighted")

    def _fusion_settings(self) -> dict[str, Any]:
        if self.config.fusion == "rrf":
            return {"method": "rrf", "rrf_k": self.config.rrf_k}
        return {"method": "weighted", "weights": list(self.config.weights),
                "normalization": self.config.normalization}
