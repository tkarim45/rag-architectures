"""The corrective retrieval loop: retrieve → grade → act (Yan et al. 2024, §3.3).

Orchestrates the three CRAG components over one query:

  1. Initial retrieval — dense top-`initial_k` over the injected CorpusIndex.
  2. Retrieval evaluator — grade every passage, aggregate to CORRECT / INCORRECT / AMBIGUOUS.
  3. Action policy:
       CORRECT   → action "refine":   keep the non-incorrect passages, knowledge-refine them.
       INCORRECT → action "fallback": discard the retrieval, rewrite the query, run the
                   broadened fallback search, refine what it finds.
       AMBIGUOUS → action "combine":  hedge — fuse the kept original passages with the
                   fallback results (RRF), then refine the fused ranking.

Fallback search — the web-search stand-in
-----------------------------------------
The paper's corrective action for INCORRECT is *web search*: leave the failing knowledge
source and look somewhere wider. This benchmark runs on a closed corpus, so the stand-in is a
wider **hybrid sweep built from core primitives**: dense ∪ BM25 over the rewritten query, fused
with Reciprocal Rank Fusion, at `fallback_k` (> `initial_k`) depth. Same intent — a broader,
differently-biased second look — with an honest ceiling: it can only find what the corpus
contains (see README for what that ceiling costs on multi-hop questions).

Everything the loop decided is written into `RetrievalResult.diagnostics`: per-passage grades
with confidences, the verdict, the action taken, the rewritten query (if any), fallback ids,
and strip kept/dropped counts.
"""
from __future__ import annotations

from typing import Any

from core import (CorpusIndex, Query, RetrievalResult, Runtime, ScoredChunk, StructuredCaller,
                  rrf)

from .config import Config
from .evaluator import Evaluation, RetrievalEvaluator
from .refiner import KnowledgeRefiner
from .rewriter import QueryRewriter

#: Action labels, fixed vocabulary — diagnostics consumers switch on these.
ACTION_REFINE = "refine"
ACTION_FALLBACK = "fallback"
ACTION_COMBINE = "combine"


class CorrectiveRetriever:
    """One instance per pipeline: owns the evaluator, refiner and rewriter, reads the index."""

    def __init__(self, runtime: Runtime, index: CorpusIndex, config: Config) -> None:
        self._index = index
        self._config = config
        self._tracer = runtime.tracer
        self._evaluator = RetrievalEvaluator(StructuredCaller(runtime.llm), config,
                                             runtime.tracer)
        self._refiner = KnowledgeRefiner(runtime.llm, config, runtime.tracer)
        self._rewriter = QueryRewriter(runtime.llm, runtime.tracer)

    def retrieve(self, question: str) -> RetrievalResult:
        cfg = self._config
        with self._tracer.span("corrective.retrieve", initial_k=cfg.initial_k) as span:
            # 1. Initial retrieval: dense only — the evaluator, not a hybrid first pass, is
            #    this architecture's defense against a bad first retrieval.
            with self._tracer.span("corrective.initial_search", k=cfg.initial_k):
                initial = self._index.dense_search(question, cfg.initial_k)

            # 2. Grade each passage; aggregate to the per-query verdict.
            evaluation = self._evaluator.evaluate(question, initial)

            # 3. Branch on the verdict.
            rewritten: str | None = None
            fallback: list[ScoredChunk] = []
            if evaluation.verdict == "correct":
                action = ACTION_REFINE
                selected = self._kept(initial, evaluation)[:cfg.final_k]
            elif evaluation.verdict == "incorrect":
                action = ACTION_FALLBACK
                rewritten = self._rewriter.rewrite(question)
                fallback = self._fallback_search(rewritten)
                selected = fallback[:cfg.final_k]
            else:  # ambiguous
                action = ACTION_COMBINE
                rewritten = self._rewriter.rewrite(question)
                fallback = self._fallback_search(rewritten)
                combined = rrf([self._kept(initial, evaluation), fallback],
                               k=cfg.rrf_k, retriever="corrective.combined")
                selected = combined[:cfg.final_k]

            # 4. Knowledge refinement on whatever the branch selected.
            refined, report = self._refiner.refine(question, selected)

            span.set("verdict", evaluation.verdict)
            span.set("action", action)
            span.set("selected", len(refined))

            query = Query(text=question, top_k=cfg.final_k,
                          variants=(rewritten,) if rewritten else ())
            return RetrievalResult(query=query, chunks=refined,
                                   diagnostics=self._diagnostics(
                                       evaluation, action, rewritten, initial, fallback,
                                       report.to_dict()))

    # ---- internals ---------------------------------------------------------------------

    @staticmethod
    def _kept(initial: list[ScoredChunk], evaluation: Evaluation) -> list[ScoredChunk]:
        """The passages worth keeping from the initial retrieval: everything not graded
        incorrect, in original rank order (correct AND ambiguous — the evaluator only has
        license to discard what it positively judged irrelevant)."""
        return [h for h in initial if evaluation.grade_of(h.chunk_id) != "incorrect"]

    def _fallback_search(self, query: str) -> list[ScoredChunk]:
        """Broadened hybrid sweep — the closed-corpus stand-in for the paper's web search:
        dense ∪ BM25 over the rewritten query, RRF-fused, at fallback_k depth."""
        cfg = self._config
        with self._tracer.span("corrective.fallback_search", k=cfg.fallback_k) as span:
            dense = self._index.dense_search(query, cfg.fallback_k)
            sparse = self._index.sparse_search(query, cfg.fallback_k)
            fused = rrf([dense, sparse], k=cfg.rrf_k, retriever="corrective.fallback")
            span.set("dense", len(dense))
            span.set("sparse", len(sparse))
            span.set("fused", len(fused))
        return fused

    @staticmethod
    def _diagnostics(evaluation: Evaluation, action: str, rewritten: str | None,
                     initial: list[ScoredChunk], fallback: list[ScoredChunk],
                     refinement: dict[str, Any]) -> dict[str, Any]:
        return {
            "architecture": "corrective",
            "verdict": evaluation.verdict,
            "action": action,
            "grades": [g.to_dict() for g in evaluation.grades],
            "rewritten_query": rewritten,
            "initial_chunk_ids": [h.chunk_id for h in initial],
            "fallback_chunk_ids": [h.chunk_id for h in fallback],
            "refinement": refinement,
        }
