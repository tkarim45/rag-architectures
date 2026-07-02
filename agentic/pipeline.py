"""Agentic RAG pipeline — one ReAct run serves both `retrieve()` and `answer()`.

Why the two entrypoints share a single agent run
------------------------------------------------
A trajectory costs up to ``max_steps`` LLM calls — by far the most expensive retrieval in this
repo. The benchmark calls ``retrieve()`` and then generates externally; an interactive caller
calls ``answer()``. If each entrypoint ran its own loop, benchmarking `retrieve()` +
inspecting `answer()` on the same question would double the spend *and* (with a real, sampled LLM)
produce two different trajectories, so the answer you read would not correspond to the evidence
that was scored. The pipeline therefore memoizes finished runs in a small FIFO-evicted dict keyed
by the question string: ``retrieve()`` executes the loop and caches; ``answer()`` reuses the
cached trajectory when present. Plain dict, tiny cap (``config.cache_size``) — this is a
cost/coherence device, not a semantic cache.

`retrieve()` is evidence-collection mode: the loop runs to whatever end it reaches (final answer,
budget, malformed action) and the ``RetrievalResult``/``ContextBlock`` are built from the
:class:`~agentic.evidence.EvidenceLog` regardless — an agent that ran out of budget mid-chain
still surfaced evidence worth scoring.

`answer()` prefers the agent's own ``final_answer`` (it was written while looking at the
observations, cited here as the evidence doc ids); if the loop ended without one, it falls back to
the shared ``core.AnswerGenerator`` over the evidence context, so the pipeline always returns a
grounded answer or an honest abstention.
"""
from __future__ import annotations

from dataclasses import dataclass

from core import (AnswerGenerator, ContextBlock, ContextBuilder, CorpusIndex, GeneratedAnswer,
                  PipelineResult, Query, RetrievalResult, Runtime, StructuredCaller)

from .agent import ReActAgent, Trajectory
from .config import Config
from .evidence import EvidenceLog
from .tools import build_default_registry


@dataclass(frozen=True)
class AgentRun:
    """Everything one trajectory produced — the cached unit shared by retrieve()/answer()."""

    trajectory: Trajectory
    retrieval: RetrievalResult
    context: ContextBlock


class Pipeline:
    """ReAct-style tool-using retrieval agent over a shared corpus index."""

    def __init__(self, runtime: Runtime, config: Config | None = None, *,
                 index: CorpusIndex | None = None) -> None:
        self.runtime = runtime
        self.config = config or Config()
        self._index = index
        self._caller = StructuredCaller(runtime.llm)
        self._context_builder = ContextBuilder(max_passages=self.config.max_context_passages,
                                               max_chars=self.config.max_context_chars)
        self._generator = AnswerGenerator(runtime.llm, tracer=runtime.tracer)
        self._runs: dict[str, AgentRun] = {}     # question → finished run, FIFO-evicted

    # ---- lazy offline artifacts ----------------------------------------------------------

    @property
    def index(self) -> CorpusIndex:
        """Injectable by the benchmark (shared artifacts ⇒ honest comparison); built lazily from
        runtime.corpus otherwise so Pipeline construction stays cheap."""
        if self._index is None:
            self._index = self.runtime.build_index(self.config.chunker)
        return self._index

    # ---- package contract ------------------------------------------------------------------

    def retrieve(self, question: str) -> tuple[RetrievalResult, ContextBlock]:
        """Run (or reuse) the agent loop and expose its evidence as a ranked retrieval.

        Evidence-collection mode: the result is built from the EvidenceLog whether or not the
        agent reached a final answer — see module docstring."""
        run, _ = self._run(question)
        return run.retrieval, run.context

    def answer(self, question: str) -> PipelineResult:
        """Full run: agent's own final answer when it produced one, generator fallback else."""
        with self.runtime.tracer.span("agentic.answer"):
            run, cached = self._run(question)
            if run.trajectory.final_answer is not None:
                answer = GeneratedAnswer(text=run.trajectory.final_answer,
                                         citations=tuple(run.retrieval.doc_ids))
            else:
                answer = self._generator.generate(question, run.context)
        return PipelineResult(
            query=run.retrieval.query, retrieval=run.retrieval, context=run.context,
            answer=answer,
            diagnostics={"architecture": "agentic",
                         "agent_answered": run.trajectory.final_answer is not None,
                         "cached_trajectory": cached,
                         "stop_reason": run.trajectory.stop_reason})

    # ---- one agent run ---------------------------------------------------------------------

    def _run(self, question: str) -> tuple[AgentRun, bool]:
        """Return the finished run for `question`, executing the loop only on a cache miss."""
        cached = self._runs.get(question)
        if cached is not None:
            return cached, True
        run = self._execute(question)
        if self.config.cache_size > 0:
            self._runs[question] = run
            while len(self._runs) > self.config.cache_size:      # FIFO: dicts keep insertion order
                self._runs.pop(next(iter(self._runs)))
        return run, False

    def _execute(self, question: str) -> AgentRun:
        config = self.config
        with self.runtime.tracer.span("agentic.retrieve", max_steps=config.max_steps) as span:
            evidence = EvidenceLog(recency_weight=config.recency_weight)
            registry = build_default_registry(self.index, evidence, config)
            agent = ReActAgent(caller=self._caller, registry=registry, evidence=evidence,
                               config=config, tracer=self.runtime.tracer)
            trajectory = agent.run(question)

            chunks = evidence.ranked_chunks(config.final_k)
            retrieval = RetrievalResult(
                query=Query(text=question, top_k=config.final_k),
                chunks=chunks,
                diagnostics={"architecture": "agentic",
                             **trajectory.to_diagnostics(),
                             "evidence": evidence.as_dicts(),
                             "evidence_size": len(evidence)})
            with self.runtime.tracer.span("agentic.context"):
                context = self._context_builder.build(chunks)
            span.set("stop_reason", trajectory.stop_reason)
            span.set("steps", len(trajectory.steps))
            span.set("docs", len(retrieval.doc_ids))
            span.set("truncated", context.truncated)
        return AgentRun(trajectory=trajectory, retrieval=retrieval, context=context)
