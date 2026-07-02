"""Hybrid RAG pipeline — orchestrates (dense ∥ sparse) → fuse → context → generate.

The pipeline owns lifecycle and wiring only; retrieval logic lives in ``HybridRetriever``. Two
entrypoints per the package contract:

* ``retrieve(question)`` — the online retrieval path the benchmark calls; no LLM generation.
* ``answer(question)``   — retrieve + ``core.AnswerGenerator``; the standalone entrypoint.

The index is injectable so the benchmark can build one ``CorpusIndex`` per chunking strategy and
share it across all architectures (identical offline artifacts ⇒ honest comparison). When omitted,
the pipeline lazily builds its own from ``runtime.corpus`` on first use — lazily so constructing a
Pipeline stays cheap and imports never trigger embedding work.
"""
from __future__ import annotations

from core import (AnswerGenerator, ContextBlock, ContextBuilder, CorpusIndex, PipelineResult,
                  Query, RetrievalResult, Runtime)

from .config import Config
from .retriever import HybridRetriever


class Pipeline:
    """Hybrid retrieval RAG over a shared corpus index."""

    def __init__(self, runtime: Runtime, config: Config | None = None, *,
                 index: CorpusIndex | None = None) -> None:
        self.runtime = runtime
        self.config = config or Config()
        self._index = index
        self._retriever: HybridRetriever | None = None
        self._context_builder = ContextBuilder(max_passages=self.config.max_context_passages,
                                               max_chars=self.config.max_context_chars)
        self._generator = AnswerGenerator(runtime.llm, tracer=runtime.tracer)

    # ---- lazy offline artifacts ----------------------------------------------------------

    @property
    def index(self) -> CorpusIndex:
        if self._index is None:
            self._index = self.runtime.build_index(self.config.chunker)
        return self._index

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(index=self.index, config=self.config,
                                              tracer=self.runtime.tracer)
        return self._retriever

    # ---- online path ---------------------------------------------------------------------

    def retrieve(self, question: str) -> tuple[RetrievalResult, ContextBlock]:
        """Online retrieval only — what the benchmark scores. No answer generation."""
        with self.runtime.tracer.span("hybrid.retrieve", fusion=self.config.fusion) as span:
            query = Query(text=question, top_k=self.config.final_k)
            result = self.retriever.retrieve(query)
            with self.runtime.tracer.span("hybrid.context"):
                context = self._context_builder.build(result.chunks)
            span.set("docs", len(result.doc_ids))
            span.set("truncated", context.truncated)
        return result, context

    def answer(self, question: str) -> PipelineResult:
        """retrieve() + grounded generation — the standalone entrypoint."""
        with self.runtime.tracer.span("hybrid.answer"):
            retrieval, context = self.retrieve(question)
            answer = self._generator.generate(question, context)
        return PipelineResult(query=retrieval.query, retrieval=retrieval, context=context,
                              answer=answer, diagnostics={"architecture": "hybrid",
                                                          **retrieval.diagnostics})
