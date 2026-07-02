"""The sparse (BM25) pipeline: analyze query → BM25 top-k → stuff context → generate.

Structurally the mirror image of ``naive``: same corpus, same chunker, same context assembly, same
grounded generator — only the retrieval half differs (inverted-index lexical scoring instead of
embedding cosine). Keeping everything else identical is what lets the benchmark read sparse-vs-
naive deltas as a pure lexical-vs-semantic comparison.
"""
from __future__ import annotations

from core import (AnswerGenerator, ContextBlock, ContextBuilder, CorpusIndex, PipelineResult,
                  Query, RetrievalResult, Runtime)

from .config import SparseConfig
from .retriever import BM25Retriever


class Pipeline:
    """Lexical RAG: BM25 over chunk text, then the shared grounded generation path."""

    def __init__(self, runtime: Runtime, config: SparseConfig | None = None, *,
                 index: CorpusIndex | None = None) -> None:
        """The benchmark injects a shared ``index``; standalone callers get a lazy build from
        ``runtime.corpus``. The retriever is also built lazily (it may need to construct a
        config-specific BM25 index, and that work belongs to first use, not construction)."""
        self.runtime = runtime
        self.config = config or SparseConfig()
        self._index = index
        self._retriever: BM25Retriever | None = None
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
    def retriever(self) -> BM25Retriever:
        if self._retriever is None:
            self._retriever = BM25Retriever(self.index, self.config, tracer=self.runtime.tracer)
        return self._retriever

    def retrieve(self, question: str) -> tuple[RetrievalResult, ContextBlock]:
        """Online retrieval path only — what the benchmark scores. Zero LLM and zero embedding
        calls: BM25's online cost is a tokenize + inverted-index scan, which is why it remains
        the production workhorse it is."""
        with self.runtime.tracer.span("sparse.retrieve", top_k=self.config.top_k) as span:
            result = self.retriever.retrieve(Query(text=question, top_k=self.config.top_k))
            with self.runtime.tracer.span("sparse.build_context"):
                context = self._context_builder.build(result.chunks)
            result.diagnostics["context_truncated"] = context.truncated
            result.diagnostics["context_passages"] = len(context.chunk_ids)
            span.set("docs", len(result.doc_ids))
        return result, context

    def answer(self, question: str) -> PipelineResult:
        """Standalone entrypoint: retrieve + grounded generation. When BM25 finds no lexical
        overlap at all, the context is empty and the generator abstains — an honest 'I don't
        know' beats an answer hallucinated over zero evidence."""
        with self.runtime.tracer.span("sparse.pipeline"):
            retrieval, context = self.retrieve(question)
            answer = self._generator.generate(question, context)
        return PipelineResult(query=retrieval.query, retrieval=retrieval, context=context,
                              answer=answer, diagnostics=dict(retrieval.diagnostics))
