"""HyDE pipeline — hypothesize → embed → mix → dense search → generate.

Online path per question:
  1. LLM writes `n_hypotheses` short hypothetical documents that would answer the question
     (facts may be invented; only their vocabulary/shape matters).
  2. Each hypothesis is embedded; the mean hypothesis vector is blended with the real query
     vector at `query_weight` and L2-renormalized.
  3. The mixed vector probes the shared dense index (`dense_search_vector`).
  4. Top chunks become the context block; `answer()` additionally runs the core generator.

The index is injected by the benchmark (so all architectures share identical offline artifacts)
or built lazily from the runtime corpus on first use.
"""
from __future__ import annotations

from core import (AnswerGenerator, ContextBlock, ContextBuilder, CorpusIndex, PipelineResult,
                  RetrievalResult, Runtime)

from .config import Config
from .hypothesis import generate_hypotheses
from .retriever import retrieve


class Pipeline:
    """HyDE (Gao et al. 2022) over the shared corpus index."""

    def __init__(self, runtime: Runtime, config: Config | None = None, *,
                 index: CorpusIndex | None = None) -> None:
        self.runtime = runtime
        self.config = config or Config()
        self._index = index
        self._context_builder = ContextBuilder(max_passages=self.config.final_k,
                                               max_chars=self.config.max_context_chars)
        self._generator = AnswerGenerator(runtime.llm, tracer=runtime.tracer)

    # ---- offline -------------------------------------------------------------------------

    @property
    def index(self) -> CorpusIndex:
        """The dense+sparse index; built once from the runtime corpus if not injected."""
        if self._index is None:
            self._index = self.runtime.build_index(self.config.chunker)
        return self._index

    # ---- online --------------------------------------------------------------------------

    def retrieve(self, question: str) -> tuple[RetrievalResult, ContextBlock]:
        """Online retrieval path only — no answer generation. This is what the benchmark calls."""
        with self.runtime.tracer.span("hyde.pipeline", question=question):
            hypotheses = generate_hypotheses(self.runtime.llm, question, self.config,
                                             self.runtime.tracer)
            result = retrieve(self.index, question, hypotheses, self.config,
                              self.runtime.tracer)
            context = self._context_builder.build(result.chunks)
        return result, context

    def answer(self, question: str) -> PipelineResult:
        """retrieve() + grounded generation — the standalone entrypoint."""
        result, context = self.retrieve(question)
        generated = self._generator.generate(question, context)
        return PipelineResult(query=result.query, retrieval=result, context=context,
                              answer=generated, diagnostics=dict(result.diagnostics))
