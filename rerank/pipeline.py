"""Rerank RAG pipeline — orchestrates candidates → cross-encoder rerank → context → generate.

The pipeline owns lifecycle and wiring only; funnel logic lives in ``RerankRetriever``. Two
entrypoints per the package contract:

* ``retrieve(question)`` — the online retrieval path the benchmark calls; no LLM generation.
* ``answer(question)``   — retrieve + ``core.AnswerGenerator``; the standalone entrypoint.

Both heavy resources are injectable and lazy:

* ``index`` — the benchmark builds one ``CorpusIndex`` per chunking strategy and shares it across
  architectures (identical offline artifacts ⇒ honest comparison). Omitted ⇒ built lazily from
  ``runtime.corpus`` on first use.
* ``reranker`` — offline tests inject ``LexicalOverlapReranker`` so the two-stage plumbing runs
  without torch/model downloads. Omitted ⇒ a ``CrossEncoderReranker`` per config, which itself
  lazy-loads its weights on first query.
"""
from __future__ import annotations

from core import (AnswerGenerator, ContextBlock, ContextBuilder, CorpusIndex, PipelineResult,
                  Query, RetrievalResult, Runtime)

from .config import Config
from .reranker import CrossEncoderReranker, Reranker
from .retriever import RerankRetriever


class Pipeline:
    """Two-stage retrieve-then-rerank RAG over a shared corpus index."""

    def __init__(self, runtime: Runtime, config: Config | None = None, *,
                 index: CorpusIndex | None = None, reranker: Reranker | None = None) -> None:
        self.runtime = runtime
        self.config = config or Config()
        self._index = index
        self._reranker = reranker
        self._retriever: RerankRetriever | None = None
        self._context_builder = ContextBuilder(max_passages=self.config.max_context_passages,
                                               max_chars=self.config.max_context_chars)
        self._generator = AnswerGenerator(runtime.llm, tracer=runtime.tracer)

    # ---- lazy resources --------------------------------------------------------------------

    @property
    def index(self) -> CorpusIndex:
        if self._index is None:
            self._index = self.runtime.build_index(self.config.chunker)
        return self._index

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.config.cross_encoder_model,
                                                  batch_size=self.config.batch_size)
        return self._reranker

    @property
    def retriever(self) -> RerankRetriever:
        if self._retriever is None:
            self._retriever = RerankRetriever(index=self.index, reranker=self.reranker,
                                              config=self.config, tracer=self.runtime.tracer)
        return self._retriever

    # ---- online path ---------------------------------------------------------------------

    def retrieve(self, question: str) -> tuple[RetrievalResult, ContextBlock]:
        """Online retrieval only — what the benchmark scores. No answer generation."""
        with self.runtime.tracer.span("rerank.retrieve",
                                      candidate_k=self.config.candidate_k) as span:
            query = Query(text=question, top_k=self.config.final_k)
            result = self.retriever.retrieve(query)
            with self.runtime.tracer.span("rerank.context"):
                context = self._context_builder.build(result.chunks)
            span.set("docs", len(result.doc_ids))
            span.set("truncated", context.truncated)
        return result, context

    def answer(self, question: str) -> PipelineResult:
        """retrieve() + grounded generation — the standalone entrypoint."""
        with self.runtime.tracer.span("rerank.answer"):
            retrieval, context = self.retrieve(question)
            answer = self._generator.generate(question, context)
        return PipelineResult(query=retrieval.query, retrieval=retrieval, context=context,
                              answer=answer, diagnostics={"architecture": "rerank",
                                                          **retrieval.diagnostics})
