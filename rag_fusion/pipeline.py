"""RAG-Fusion pipeline: broaden → parallel dense fan-out → Reciprocal Rank Fusion → generate.

Implements Rackauckas (2024), "RAG-Fusion: a New Take on Retrieval-Augmented Generation", from
`core` primitives. The pipeline composes the package's parts (expander, fan-out retriever, RRF
fusion) with core services and writes the architecture's full story — generated queries, drops,
per-query hit counts, fusion parameters — into ``RetrievalResult.diagnostics`` so benchmarks and
traces can audit *how* a document was found, not just that it was.
"""
from __future__ import annotations

from typing import Any

from core import (AnswerGenerator, ContextBlock, ContextBuilder, CorpusIndex, PipelineResult,
                  Query, RetrievalResult, Runtime)

from .config import Config
from .expander import Expansion, QueryExpander
from .retriever import fuse, retrieve_per_query


class Pipeline:
    """RAG-Fusion.

    ``index`` may be injected by the benchmark so every architecture retrieves over identical
    offline artifacts; when omitted, an index is built lazily from ``runtime.corpus`` on first
    use with the configured chunker.
    """

    def __init__(self, runtime: Runtime, config: Config | None = None, *,
                 index: CorpusIndex | None = None) -> None:
        self.config = config or Config()
        self._runtime = runtime
        self._index = index
        self._expander = QueryExpander(
            runtime.llm, runtime.embedder, runtime.tracer,
            n_queries=self.config.n_queries,
            dedup_threshold=self.config.dedup_threshold,
            max_tokens=self.config.expansion_max_tokens)
        self._contexts = ContextBuilder(max_passages=self.config.max_context_passages,
                                        max_chars=self.config.max_context_chars)
        self._generator = AnswerGenerator(runtime.llm, tracer=runtime.tracer)

    # ---- retrieval ---------------------------------------------------------------------

    def retrieve(self, question: str) -> tuple[RetrievalResult, ContextBlock]:
        """Online retrieval path only — no answer generation (what the benchmark calls)."""
        cfg = self.config
        index = self._ensure_index()
        with self._runtime.tracer.span("rag_fusion.retrieve", final_k=cfg.final_k) as span:
            expansion = self._expander.expand(question)
            rankings = retrieve_per_query(index, expansion.queries, k=cfg.per_query_k,
                                          max_workers=cfg.max_workers,
                                          tracer=self._runtime.tracer)
            fused = fuse(rankings, rrf_k=cfg.rrf_k, tracer=self._runtime.tracer)
            selected = fused[: cfg.final_k]
            span.set("queries_searched", len(expansion.queries))
            span.set("chunks_returned", len(selected))
        query = Query(text=question, top_k=cfg.final_k, variants=tuple(expansion.queries[1:]))
        retrieval = RetrievalResult(
            query=query, chunks=selected,
            diagnostics=self._diagnostics(expansion, rankings, fused_pool=len(fused)))
        return retrieval, self._contexts.build(selected)

    def answer(self, question: str) -> PipelineResult:
        """retrieve() + grounded generation — the standalone entrypoint."""
        with self._runtime.tracer.span("rag_fusion.pipeline"):
            retrieval, context = self.retrieve(question)
            answer = self._generator.generate(question, context)
        return PipelineResult(query=retrieval.query, retrieval=retrieval, context=context,
                              answer=answer, diagnostics=dict(retrieval.diagnostics))

    # ---- internals ---------------------------------------------------------------------

    def _ensure_index(self) -> CorpusIndex:
        if self._index is None:
            self._index = self._runtime.build_index(self.config.chunker)
        return self._index

    def _diagnostics(self, expansion: Expansion, rankings: list[list[Any]], *,
                     fused_pool: int) -> dict[str, Any]:
        return {
            "generated_queries": list(expansion.queries),
            "llm_variants": list(expansion.llm_variants),
            "dropped_near_duplicates": list(expansion.dropped),
            "expansion_fallback": expansion.fallback,
            "per_query_hits": [len(r) for r in rankings],
            "fused_pool_size": fused_pool,
            "merge_strategy": "reciprocal_rank_fusion",
            "rrf_k": self.config.rrf_k,
        }
