"""Multi-query RAG pipeline: expand → parallel dense fan-out → round-robin merge → generate.

The pipeline composes the package's three parts (expander, fan-out retriever, interleave merge)
with core services, and writes the architecture's full story — generated queries, drops,
per-variant hit counts, merge strategy — into ``RetrievalResult.diagnostics`` so benchmarks and
traces can audit *how* a document was found, not just that it was.
"""
from __future__ import annotations

from typing import Any

from core import (AnswerGenerator, ContextBlock, ContextBuilder, CorpusIndex, PipelineResult,
                  Query, RetrievalResult, Runtime)

from .config import Config
from .expander import Expansion, QueryExpander
from .retriever import interleave, retrieve_per_query


class Pipeline:
    """Multi-query RAG.

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
        with self._runtime.tracer.span("multi_query.retrieve", final_k=cfg.final_k) as span:
            expansion = self._expander.expand(question)
            rankings = retrieve_per_query(index, expansion.queries, k=cfg.per_query_k,
                                          max_workers=cfg.max_workers,
                                          tracer=self._runtime.tracer)
            merged = interleave(rankings)[: cfg.final_k]
            span.set("queries_searched", len(expansion.queries))
            span.set("chunks_returned", len(merged))
        query = Query(text=question, top_k=cfg.final_k, variants=tuple(expansion.queries[1:]))
        retrieval = RetrievalResult(query=query, chunks=merged,
                                    diagnostics=self._diagnostics(expansion, rankings))
        return retrieval, self._contexts.build(merged)

    def answer(self, question: str) -> PipelineResult:
        """retrieve() + grounded generation — the standalone entrypoint."""
        with self._runtime.tracer.span("multi_query.pipeline"):
            retrieval, context = self.retrieve(question)
            answer = self._generator.generate(question, context)
        return PipelineResult(query=retrieval.query, retrieval=retrieval, context=context,
                              answer=answer, diagnostics=dict(retrieval.diagnostics))

    # ---- internals ---------------------------------------------------------------------

    def _ensure_index(self) -> CorpusIndex:
        if self._index is None:
            self._index = self._runtime.build_index(self.config.chunker)
        return self._index

    @staticmethod
    def _diagnostics(expansion: Expansion, rankings: list[list[Any]]) -> dict[str, Any]:
        return {
            "generated_queries": list(expansion.queries),
            "llm_variants": list(expansion.llm_variants),
            "dropped_near_duplicates": list(expansion.dropped),
            "expansion_fallback": expansion.fallback,
            "per_query_hits": [len(r) for r in rankings],
            "merge_strategy": "round_robin_interleave",
        }
