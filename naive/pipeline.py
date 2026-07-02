"""The naive dense-RAG pipeline: retrieve top-k → stuff context → generate.

This is the baseline every other architecture in this repo is measured against. It makes zero
online decisions — no query rewriting, no fusion, no grading, no second pass — so any score another
architecture posts above (or below) this one is attributable to its added machinery, not to a
different corpus, chunker, embedder, or generator. Those are all shared via the injected
``CorpusIndex`` and ``Runtime``.
"""
from __future__ import annotations

from core import (AnswerGenerator, ContextBlock, ContextBuilder, CorpusIndex, PipelineResult,
                  Query, RetrievalResult, Runtime)

from .config import NaiveConfig
from .retriever import DenseRetriever


class Pipeline:
    """Canonical dense RAG: embed query → top-k cosine → context → grounded answer."""

    def __init__(self, runtime: Runtime, config: NaiveConfig | None = None, *,
                 index: CorpusIndex | None = None) -> None:
        """The benchmark injects a shared ``index`` so every architecture retrieves against
        identical offline artifacts; standalone callers omit it and the pipeline builds one
        lazily from ``runtime.corpus`` on first use (index builds are expensive — don't pay
        for them at construction time)."""
        self.runtime = runtime
        self.config = config or NaiveConfig()
        self._index = index
        self._context_builder = ContextBuilder(
            max_passages=self.config.context_max_passages,
            max_chars=self.config.context_max_chars)
        self._generator = AnswerGenerator(runtime.llm, tracer=runtime.tracer)

    @property
    def index(self) -> CorpusIndex:
        if self._index is None:
            self._index = self.runtime.build_index(self.config.chunker)
        return self._index

    def retrieve(self, question: str) -> tuple[RetrievalResult, ContextBlock]:
        """Online retrieval path only — what the benchmark scores. No LLM call happens here;
        naive RAG spends its entire online budget on one embedding lookup."""
        with self.runtime.tracer.span("naive.retrieve", top_k=self.config.top_k) as span:
            retriever = DenseRetriever(index=self.index, tracer=self.runtime.tracer)
            result = retriever.retrieve(Query(text=question, top_k=self.config.top_k))
            with self.runtime.tracer.span("naive.build_context"):
                context = self._context_builder.build(result.chunks)
            result.diagnostics["context_truncated"] = context.truncated
            result.diagnostics["context_passages"] = len(context.chunk_ids)
            span.set("docs", len(result.doc_ids))
        return result, context

    def answer(self, question: str) -> PipelineResult:
        """Standalone entrypoint: retrieve + grounded generation. Generation is shared core
        machinery (strict answer-from-context-or-abstain), so a wrong answer from this pipeline
        is a retrieval failure by construction."""
        with self.runtime.tracer.span("naive.pipeline"):
            retrieval, context = self.retrieve(question)
            answer = self._generator.generate(question, context)
        return PipelineResult(query=retrieval.query, retrieval=retrieval, context=context,
                              answer=answer, diagnostics=dict(retrieval.diagnostics))
