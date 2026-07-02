"""GraphRAG pipeline — the package's public entrypoint, wired to the framework contract.

Offline/online split, made explicit:

  * OFFLINE (once): `build_graph` extracts entities/relations per document, merges them into the
    `KnowledgeGraph`, detects Louvain communities and writes their summaries. The benchmark builds
    this once and injects it into every Pipeline; standalone callers get a lazy build on first use.
  * ONLINE (per query): route → local traversal or global map-reduce → ranked provenance docs →
    whole-doc chunks → context → (optionally) a grounded answer.

Both heavyweight artifacts are injectable and lazy: `graph` (the knowledge graph) and `index`
(a whole-document `CorpusIndex`, used purely to resolve doc ids into generator-ready chunks —
GraphRAG never vector-searches it).
"""
from __future__ import annotations

from core import (AnswerGenerator, ContextBlock, ContextBuilder, CorpusIndex, PipelineResult,
                  RetrievalResult, Runtime)

from .config import Config
from .graph import KnowledgeGraph, build_graph
from .retriever import GraphRetriever


class Pipeline:
    """Standalone GraphRAG over the runtime's corpus. `retrieve` is what the benchmark scores;
    `answer` adds grounded generation for interactive use."""

    def __init__(self, runtime: Runtime, config: Config | None = None, *,
                 index: CorpusIndex | None = None, graph: KnowledgeGraph | None = None):
        self.runtime = runtime
        self.config = config or Config()
        self._index = index
        self._graph = graph
        self._retriever: GraphRetriever | None = None
        self._context_builder = ContextBuilder(max_passages=self.config.max_context_passages,
                                               max_chars=self.config.max_context_chars)
        self._generator = AnswerGenerator(runtime.llm, tracer=runtime.tracer)

    # ---- lazy offline artifacts ----------------------------------------------------------

    @property
    def graph(self) -> KnowledgeGraph:
        """The knowledge graph; built from `runtime.corpus` on first use when not injected."""
        if self._graph is None:
            self._graph = build_graph(self.runtime, self.runtime.corpus, self.config)
        return self._graph

    @property
    def index(self) -> CorpusIndex:
        """Whole-document index for doc-id → chunk resolution; built lazily when not injected."""
        if self._index is None:
            self._index = self.runtime.build_index("whole")
        return self._index

    @property
    def retriever(self) -> GraphRetriever:
        if self._retriever is None:
            self._retriever = GraphRetriever(llm=self.runtime.llm, graph=self.graph,
                                             index=self.index, config=self.config,
                                             tracer=self.runtime.tracer)
        return self._retriever

    # ---- online path ----------------------------------------------------------------------

    def retrieve(self, question: str) -> tuple[RetrievalResult, ContextBlock]:
        """Retrieval only — no generation. This is the benchmark's call."""
        with self.runtime.tracer.span("graphrag.pipeline.retrieve") as span:
            retrieval = self.retriever.retrieve(question)
            context = self._context_builder.build(retrieval.chunks)
            span.set("docs", len(retrieval.doc_ids))
            span.set("context_chars", len(context.text))
            span.set("truncated", context.truncated)
        return retrieval, context

    def answer(self, question: str) -> PipelineResult:
        """retrieve() + grounded generation — the standalone entrypoint."""
        with self.runtime.tracer.span("graphrag.pipeline.answer") as span:
            retrieval, context = self.retrieve(question)
            generated = self._generator.generate(question, context)
            span.set("abstained", generated.abstained)
        return PipelineResult(query=retrieval.query, retrieval=retrieval, context=context,
                              answer=generated, diagnostics=dict(retrieval.diagnostics))
