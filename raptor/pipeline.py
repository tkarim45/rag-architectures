"""RAPTOR pipeline — the package's contract surface.

Offline/online split:
  * OFFLINE — `build_tree` (see `tree.py`) embeds leaves and recursively clusters + summarizes.
    The benchmark calls it once and injects the tree into every Pipeline; standalone usage lets
    the pipeline build it lazily from `runtime.corpus` on first query.
  * ONLINE  — collapsed-tree scoring over all nodes (see `retriever.py`), context assembly via
    `core.ContextBuilder` (whose display-text dedup collapses the per-doc synthetic chunks back
    into unique passages), and answer generation via `core.AnswerGenerator`.
"""
from __future__ import annotations

from core import (AnswerGenerator, ContextBlock, ContextBuilder, PipelineResult,
                  RetrievalResult, Runtime)

from .config import Config
from .retriever import CollapsedTreeRetriever
from .tree import RaptorTree, build_tree


class Pipeline:
    """RAPTOR retrieve/answer pipeline over a (possibly injected) RaptorTree."""

    def __init__(self, runtime: Runtime, config: Config | None = None, *,
                 tree: RaptorTree | None = None) -> None:
        """`tree` is the benchmark injection seam: pass a prebuilt tree so all pipelines share
        one offline artifact. When omitted, the tree is built lazily from `runtime.corpus` on
        the first retrieve — never at construction, so importing/instantiating stays cheap."""
        self.runtime = runtime
        self.config = config or Config()
        self._tree = tree
        self._context_builder = ContextBuilder(
            max_passages=self.config.max_context_passages,
            max_chars=self.config.max_context_tokens * 4)     # same chars≈tokens×4 estimate
        self._generator = AnswerGenerator(
            llm=runtime.llm, max_tokens=self.config.answer_max_tokens, tracer=runtime.tracer)

    @property
    def tree(self) -> RaptorTree:
        """The offline artifact, built on first access when not injected."""
        if self._tree is None:
            self._tree = build_tree(self.runtime, self.runtime.corpus, self.config)
        return self._tree

    def retrieve(self, question: str) -> tuple[RetrievalResult, ContextBlock]:
        """Online retrieval only — what the benchmark calls. No answer generation."""
        with self.runtime.tracer.span("raptor.pipeline.retrieve"):
            retriever = CollapsedTreeRetriever(
                tree=self.tree, embedder=self.runtime.embedder,
                config=self.config, tracer=self.runtime.tracer)
            result = retriever.retrieve(question)
            context = self._context_builder.build(result.chunks)
            result.diagnostics["context_passages"] = len(context.chunk_ids)
            result.diagnostics["context_truncated"] = context.truncated
        return result, context

    def answer(self, question: str) -> PipelineResult:
        """retrieve() + AnswerGenerator — the standalone entrypoint."""
        with self.runtime.tracer.span("raptor.pipeline.answer"):
            retrieval, context = self.retrieve(question)
            generated = self._generator.generate(question, context)
        return PipelineResult(
            query=retrieval.query, retrieval=retrieval, context=context, answer=generated,
            diagnostics={"architecture": "raptor",
                         "tree_shape": self.tree.describe(),
                         **retrieval.diagnostics})
