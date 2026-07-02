"""RerankRetriever — orchestrates the recall-then-precision funnel.

Stage 1 (recall): dense search over-fetches ``candidate_k`` chunks; optionally a BM25 branch is
unioned in (dedup by chunk id, dense order first) so exact-token matches embeddings miss still
reach the reranker. Union rather than fusion, deliberately: stage-1 order is irrelevant because
stage 2 re-scores every candidate from scratch — fusing here would add a hyperparameter that
cannot affect the output ranking.

Stage 2 (precision): the injected ``Reranker`` re-scores all candidates against the query and
re-sorts. An optional ``score_threshold`` then drops confidently-irrelevant survivors, and the
list is cut to ``final_k``.

Diagnostics record the funnel's story — candidate counts per branch, and *rank movement*: where
the post-rerank top results originally sat in the stage-1 list. Large movements are the reranker
earning its latency; zero movement means stage 1 already had the order right and the
cross-encoder is pure overhead for this workload.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core import CorpusIndex, Query, RetrievalResult, ScoredChunk, Tracer, tracer

from .config import Config
from .reranker import Reranker


@dataclass
class RerankRetriever:
    """Two-stage retrieval: high-recall candidate generation → precision reranking. Stateless
    between queries; index and reranker are injected (the benchmark shares the index across
    architectures and injects an offline reranker in tests)."""

    index: CorpusIndex
    reranker: Reranker
    config: Config = field(default_factory=Config)
    tracer: Tracer = field(default_factory=lambda: tracer)

    def retrieve(self, query: Query) -> RetrievalResult:
        cfg = self.config

        # ---- stage 1: recall ---------------------------------------------------------------
        with self.tracer.span("rerank.candidates", k=cfg.candidate_k,
                              sparse=cfg.use_sparse_candidates) as span:
            dense = self.index.dense_search(query.text, cfg.candidate_k)
            sparse: list[ScoredChunk] = []
            if cfg.use_sparse_candidates:
                sparse = self.index.sparse_search(query.text, cfg.candidate_k)
            candidates = self._union(dense, sparse)
            span.set("dense_hits", len(dense))
            span.set("sparse_hits", len(sparse))
            span.set("candidates", len(candidates))

        # ---- stage 2: precision --------------------------------------------------------------
        with self.tracer.span("rerank.rerank", reranker=self.reranker.name,
                              candidates=len(candidates)) as span:
            reranked = self.reranker.rerank(query.text, candidates)
            kept = reranked
            if cfg.score_threshold is not None:
                kept = [hit for hit in reranked if hit.score >= cfg.score_threshold]
            final = kept[:cfg.final_k]
            span.set("dropped_by_threshold", len(reranked) - len(kept))
            span.set("kept", len(final))

        diagnostics = self._diagnostics(dense, sparse, candidates, reranked, kept, final)
        return RetrievalResult(query=query, chunks=final, diagnostics=diagnostics)

    # ------------------------------------------------------------------------------------

    @staticmethod
    def _union(dense: list[ScoredChunk], sparse: list[ScoredChunk]) -> list[ScoredChunk]:
        """Dedup union, dense hits first. Order only matters for tie-breaking and rank-movement
        accounting — the reranker re-scores everything regardless."""
        seen: set[str] = set()
        merged: list[ScoredChunk] = []
        for hit in [*dense, *sparse]:
            if hit.chunk_id not in seen:
                seen.add(hit.chunk_id)
                merged.append(hit)
        return merged

    def _diagnostics(self, dense: list[ScoredChunk], sparse: list[ScoredChunk],
                     candidates: list[ScoredChunk], reranked: list[ScoredChunk],
                     kept: list[ScoredChunk], final: list[ScoredChunk]) -> dict[str, Any]:
        """The funnel's audit trail: how wide stage 1 cast, and how far stage 2 moved things."""
        stage1_rank = {hit.chunk_id: rank for rank, hit in enumerate(candidates)}
        # rank movement: stage-1 position minus final position, per surviving chunk.
        # positive = promoted by the reranker; 0 everywhere = reranker changed nothing.
        movements = [{"chunk_id": hit.chunk_id,
                      "stage1_rank": stage1_rank[hit.chunk_id],
                      "final_rank": rank,
                      "moved": stage1_rank[hit.chunk_id] - rank}
                     for rank, hit in enumerate(final)]
        return {
            "reranker": self.reranker.name,
            "candidates": {
                "total": len(candidates),
                "dense": len(dense),
                "sparse": len(sparse),
                "overlap": len(dense) + len(sparse) - len(candidates),
            },
            "score_threshold": self.config.score_threshold,
            "dropped_by_threshold": len(reranked) - len(kept),
            "rank_movement": movements,
            # how deep in the stage-1 list the final top-1 was hiding; a large value is the
            # reranker's headline win (and evidence candidate_k must stay ≥ that deep).
            "top1_from_stage1_rank": movements[0]["stage1_rank"] if movements else None,
            "final_k": self.config.final_k,
        }
