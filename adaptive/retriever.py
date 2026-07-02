"""The adaptive dispatcher: classify → route → execute → assemble one diagnostics story.

This is the seam where Adaptive-RAG's two halves meet. The classifier produces a label; this
module maps labels to executable routes and merges the route's outcome with the classifier's
verdict into a single ``RetrievalResult`` whose diagnostics tell the whole story: which route
ran and *why*, what sub-queries the multi-step loop issued, what each iteration newly found,
and why the loop stopped. The benchmark reads nothing but this record, so every online decision
the architecture makes must be legible here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core import CorpusIndex, Query, RetrievalResult, Runtime, ScoredChunk, StructuredCaller

from .classifier import Classification, ComplexityClassifier, Label
from .config import AdaptiveConfig
from .strategies import MultiStepRetriever, RouteOutcome, no_retrieval, single_step

#: Human-readable route names, keyed by classifier label (the paper's A/B/C).
ROUTE_NAMES: dict[str, str] = {"A": "no_retrieval", "B": "single_step", "C": "multi_step"}


@dataclass
class AdaptiveRetriever:
    """Online retrieval for Adaptive-RAG over a prebuilt :class:`CorpusIndex`.

    Holds no per-query state; the classifier and the multi-step loop are constructed once and
    reused, so the only per-call cost is the routing decision plus whatever the chosen route
    spends — which is the architecture's entire value proposition.
    """

    runtime: Runtime
    index: CorpusIndex
    config: AdaptiveConfig

    def __post_init__(self) -> None:
        self._classifier = ComplexityClassifier(self.runtime.llm, self.config,
                                                tracer=self.runtime.tracer)
        self._multi_step = MultiStepRetriever(self.index, StructuredCaller(self.runtime.llm),
                                              self.config, tracer=self.runtime.tracer)

    def retrieve(self, question: str) -> RetrievalResult:
        """Classify the question, run the routed strategy, and return evidence capped at
        ``config.final_k`` with full routing diagnostics."""
        classification = self._classifier.classify(question)
        route_name = ROUTE_NAMES[classification.label]

        with self.runtime.tracer.span(f"adaptive.route.{route_name}",
                                      label=classification.label) as span:
            outcome = self._dispatch(classification.label, question)
            chunks = outcome.chunks[:self.config.final_k]
            span.set("evidence_chunks", len(outcome.chunks))
            span.set("kept_chunks", len(chunks))

        query = Query(text=question, top_k=self._route_budget(classification.label),
                      variants=outcome.sub_queries)
        return RetrievalResult(query=query, chunks=chunks,
                               diagnostics=self._diagnostics(classification, route_name,
                                                             outcome, chunks))

    # ---- internals -----------------------------------------------------------------------

    def _dispatch(self, label: Label, question: str) -> RouteOutcome:
        if label == "A":
            return no_retrieval(question)
        if label == "B":
            return single_step(question, self.index, self.config, tracer=self.runtime.tracer)
        return self._multi_step.retrieve(question)

    def _route_budget(self, label: Label) -> int:
        """The k recorded on the Query — what the routed strategy was actually allowed."""
        if label == "A":
            return 0
        if label == "B":
            return self.config.single_k
        return self.config.final_k

    def _diagnostics(self, classification: Classification, route_name: str,
                     outcome: RouteOutcome, kept: list[ScoredChunk]) -> dict[str, Any]:
        diag: dict[str, Any] = {
            "architecture": "adaptive",
            "route": classification.label,
            "route_name": route_name,
            "classifier": {
                "label": classification.label,
                "raw_label": classification.raw_label,
                "reason": classification.reason,
                "coerced": classification.coerced,
                "fallback": classification.fallback,
            },
            "sub_queries": list(outcome.sub_queries),
            "evidence_chunks": len(outcome.chunks),
            "kept_chunks": len(kept),
        }
        diag.update(outcome.diagnostics)  # route-specific: seed/iterations/stop_reason or scores
        return diag
