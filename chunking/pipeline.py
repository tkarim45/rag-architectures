"""Chunking-strategy RAG pipeline.

Thesis: retrieval quality is often decided at *index* time, not query time. The same dense
retriever over the same corpus swings materially depending on how the corpus was chunked. So this
pipeline is naive RAG with one substitution — the index it reads was built by a named chunking
strategy (`sentence_window`, `parent_child`, `contextual`, or any other core `CHUNKER_REGISTRY`
name). The online path never changes; the benchmark instantiates one pipeline per strategy and
compares them on identical queries.

The offline/online split mirrors production: index construction (chunk → embed → store) happens
once via `runtime.build_index` — or is injected pre-built by the benchmark so all strategies share
identical embedder/LLM wiring — and the query path only reads.
"""
from __future__ import annotations

from core import (AnswerGenerator, CHUNKER_REGISTRY, ConfigurationError, ContextBlock,
                  ContextBuilder, CorpusIndex, PipelineResult, Query, RetrievalResult, Runtime)

from .config import Config
from .retriever import DenseStrategyRetriever


class Pipeline:
    """One chunking strategy, benchmark-comparable.

    `retrieve()` is the benchmark surface (no generation); `answer()` is the standalone
    entrypoint (retrieve + grounded generation).
    """

    def __init__(self, runtime: Runtime, config: Config | None = None, *,
                 index: CorpusIndex | None = None,
                 strategy: str = "sentence_window") -> None:
        """`index` lets the benchmark inject a pre-built index so every architecture shares
        identical offline artifacts; when omitted, the index is built lazily from
        `runtime.corpus` on first use (contextual chunking spends its per-document LLM calls
        at that moment, not at construction)."""
        if strategy not in CHUNKER_REGISTRY:
            raise ConfigurationError(
                f"unknown chunking strategy {strategy!r}; known: {sorted(CHUNKER_REGISTRY)}")
        if index is not None and index.strategy and index.strategy != strategy:
            raise ConfigurationError(
                f"injected index was built with strategy {index.strategy!r} but the pipeline "
                f"was asked for {strategy!r} — the benchmark's comparison would be mislabeled")
        self.runtime = runtime
        self.config = config or Config()
        self.strategy = strategy
        self._index = index
        self._retriever: DenseStrategyRetriever | None = None
        self._context_builder = ContextBuilder(max_passages=self.config.final_k,
                                               max_chars=self.config.max_context_chars)
        self._generator = AnswerGenerator(runtime.llm, tracer=runtime.tracer)

    # ---- offline artifacts (lazy) --------------------------------------------------------

    @property
    def index(self) -> CorpusIndex:
        if self._index is None:
            self._index = self.runtime.build_index(
                self.strategy, **self.config.chunker_kwargs(self.strategy))
        return self._index

    @property
    def retriever(self) -> DenseStrategyRetriever:
        if self._retriever is None:
            self._retriever = DenseStrategyRetriever(
                index=self.index, config=self.config, tracer=self.runtime.tracer)
        return self._retriever

    # ---- online path ---------------------------------------------------------------------

    def retrieve(self, question: str) -> tuple[RetrievalResult, ContextBlock]:
        """Online retrieval only — what the benchmark calls and scores."""
        query = Query(text=question, top_k=self.config.top_k)
        with self.runtime.tracer.span("chunking.pipeline", strategy=self.strategy) as span:
            retrieval = self.retriever.retrieve(query)
            with self.runtime.tracer.span("chunking.build_context",
                                          candidates=len(retrieval.chunks)) as ctx_span:
                context = self._context_builder.build(retrieval.chunks)
                ctx_span.set("passages", len(context.chunk_ids))
                ctx_span.set("truncated", context.truncated)
            # extend the retriever's story with what actually reached the generator — this is
            # where parent_child's dedup-and-token-cost mechanics become measurable
            retrieval.diagnostics.update({
                "context_passages": len(context.chunk_ids),
                "context_chars": len(context.text),
                "context_truncated": context.truncated,
            })
            span.set("docs", len(retrieval.doc_ids))
        return retrieval, context

    def answer(self, question: str) -> PipelineResult:
        """retrieve() + grounded generation — the standalone entrypoint."""
        retrieval, context = self.retrieve(question)
        answer = self._generator.generate(question, context)
        return PipelineResult(query=retrieval.query, retrieval=retrieval, context=context,
                              answer=answer, diagnostics={"strategy": self.strategy})
