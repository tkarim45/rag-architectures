"""Corrective RAG pipeline — the package's public entrypoint.

Thin by design: all corrective behavior lives in `retriever.py`; this class only wires the
runtime, owns the (possibly injected) index, builds the answer context, and generates. The
benchmark calls `retrieve()` with a shared pre-built index; standalone callers use `answer()`
and get a lazily built index from `runtime.corpus` on first use.
"""
from __future__ import annotations

from core import (AnswerGenerator, ContextBlock, ContextBuilder, CorpusIndex, PipelineResult,
                  RetrievalResult, Runtime)

from .config import Config
from .retriever import CorrectiveRetriever


class Pipeline:
    """CRAG over a shared CorpusIndex: retrieve → grade → act → refine → answer."""

    def __init__(self, runtime: Runtime, config: Config | None = None, *,
                 index: CorpusIndex | None = None) -> None:
        """`index` may be injected by the benchmark so every architecture retrieves against
        identical offline artifacts; when omitted, one is built lazily from `runtime.corpus`
        with the configured chunker on first use."""
        self._runtime = runtime
        self._config = config or Config()
        self._index = index
        self._retriever: CorrectiveRetriever | None = None
        self._context_builder = ContextBuilder(max_passages=self._config.max_context_passages,
                                               max_chars=self._config.max_context_chars)
        self._generator = AnswerGenerator(runtime.llm, tracer=runtime.tracer)

    @property
    def config(self) -> Config:
        return self._config

    def retrieve(self, question: str) -> tuple[RetrievalResult, ContextBlock]:
        """Online retrieval path only — no answer generation. This is what the benchmark calls."""
        with self._runtime.tracer.span("corrective.pipeline", question=question):
            result = self._get_retriever().retrieve(question)
            context = self._context_builder.build(result.chunks)
        return result, context

    def answer(self, question: str) -> PipelineResult:
        """retrieve() + AnswerGenerator — the standalone entrypoint."""
        result, context = self.retrieve(question)
        answer = self._generator.generate(question, context)
        return PipelineResult(query=result.query, retrieval=result, context=context,
                              answer=answer, diagnostics=dict(result.diagnostics))

    # ---- internals ---------------------------------------------------------------------

    def _get_retriever(self) -> CorrectiveRetriever:
        if self._retriever is None:
            if self._index is None:
                self._index = self._runtime.build_index(self._config.chunker)
            self._retriever = CorrectiveRetriever(self._runtime, self._index, self._config)
        return self._retriever
