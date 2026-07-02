"""The Adaptive-RAG pipeline: classify complexity → cheapest sufficient retrieval → generate.

Wires the classifier-routed retriever to the shared core context builder and grounded generator.
Everything expensive and shareable (index) is injected or built lazily; everything decisional
(routing, iteration) happens inside :class:`AdaptiveRetriever` per query. Because generation is
the same strict answer-from-context-or-abstain machinery every architecture uses, a wrong answer
from this pipeline is attributable to routing or retrieval — never to a different generator.
"""
from __future__ import annotations

from core import (AnswerGenerator, ContextBlock, ContextBuilder, CorpusIndex, PipelineResult,
                  RetrievalResult, Runtime)

from .config import AdaptiveConfig
from .retriever import AdaptiveRetriever


class Pipeline:
    """Adaptive-RAG (Jeong et al. 2024): route each query by predicted complexity to the
    cheapest strategy expected to answer it — no retrieval, one dense pass, or an iterative
    fused retrieval chain."""

    def __init__(self, runtime: Runtime, config: AdaptiveConfig | None = None, *,
                 index: CorpusIndex | None = None) -> None:
        """The benchmark injects a shared ``index`` so every architecture retrieves against
        identical offline artifacts; standalone callers omit it and the pipeline builds one
        lazily from ``runtime.corpus`` on first use (index builds are expensive — don't pay
        for them at construction time)."""
        self.runtime = runtime
        self.config = config or AdaptiveConfig()
        self._index = index
        self._retriever: AdaptiveRetriever | None = None
        self._context_builder = ContextBuilder(
            max_passages=self.config.context_max_passages,
            max_chars=self.config.context_max_chars)
        self._generator = AnswerGenerator(runtime.llm, tracer=runtime.tracer)

    @property
    def index(self) -> CorpusIndex:
        if self._index is None:
            self._index = self.runtime.build_index(self.config.chunker)
        return self._index

    @property
    def retriever(self) -> AdaptiveRetriever:
        if self._retriever is None:
            self._retriever = AdaptiveRetriever(runtime=self.runtime, index=self.index,
                                                config=self.config)
        return self._retriever

    def retrieve(self, question: str) -> tuple[RetrievalResult, ContextBlock]:
        """Online retrieval path only — what the benchmark scores. Cost is route-dependent by
        design: one classifier call plus zero (A), zero (B), or 1..max_iterations (C) further
        LLM calls before any generation happens."""
        with self.runtime.tracer.span("adaptive.retrieve") as span:
            result = self.retriever.retrieve(question)
            with self.runtime.tracer.span("adaptive.build_context"):
                context = self._context_builder.build(result.chunks)
            result.diagnostics["context_truncated"] = context.truncated
            result.diagnostics["context_passages"] = len(context.chunk_ids)
            span.set("route", result.diagnostics["route"])
            span.set("docs", len(result.doc_ids))
        return result, context

    def answer(self, question: str) -> PipelineResult:
        """Standalone entrypoint: routed retrieval + grounded generation. On the (normally
        disabled) no-retrieval route the context is empty and the generator abstains — the
        honest outcome on a corpus whose facts no model has ever seen."""
        with self.runtime.tracer.span("adaptive.pipeline"):
            retrieval, context = self.retrieve(question)
            answer = self._generator.generate(question, context)
        return PipelineResult(query=retrieval.query, retrieval=retrieval, context=context,
                              answer=answer, diagnostics=dict(retrieval.diagnostics))
