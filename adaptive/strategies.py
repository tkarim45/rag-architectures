"""The three executable routes of Adaptive-RAG — one per complexity label.

* ``no_retrieval`` (A) — return zero evidence. Downstream, the strictly grounded core generator
  receives an empty context and abstains honestly rather than free-styling from parametric
  memory; over an open-world corpus a caller could swap in a direct LLM answer here.
* ``single_step`` (B) — one dense top-k pass, deliberately identical in shape to the naive
  baseline. This is the cheap path the classifier should send most traffic down.
* ``MultiStepRetriever`` (C) — the paper's iterative retrieve-and-read chain: accumulate
  evidence, and each iteration ask the LLM whether the evidence now answers the question or
  what follow-up query to issue next; retrieve that follow-up with dense+BM25 RRF fusion,
  append only *new* chunks, repeat until done / stalled / out of budget.

Each route returns a ``RouteOutcome`` — chunks plus that route's diagnostics story — and stays
ignorant of routing: the classifier decides, ``retriever.py`` dispatches, routes just execute.
All three are built purely from core primitives (``CorpusIndex`` search + ``core.rrf``); this
package imports no other architecture package.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from core import CorpusIndex, ScoredChunk, StructuredCaller, Tracer, rrf
from core import tracer as default_tracer

from .config import AdaptiveConfig
from .prompts import FOLLOW_UP_PROMPT


@dataclass(frozen=True)
class RouteOutcome:
    """What a route hands back to the dispatcher: ranked evidence, the sub-queries it issued
    (beyond the original question), and its route-specific diagnostics."""

    chunks: list[ScoredChunk]
    sub_queries: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# Route A — no retrieval
# --------------------------------------------------------------------------------------------

def no_retrieval(question: str) -> RouteOutcome:
    """The paper's A route: the classifier judged the question answerable without the corpus.

    We return empty evidence and let the strictly grounded generator abstain — on this repo's
    closed fictional corpus a parametric answer is wrong by construction, which is exactly why
    ``AdaptiveConfig.allow_no_retrieval`` defaults to False and this route is normally coerced
    away before it runs. It is implemented (rather than raising) because the architecture is
    general and the benchmark should be able to measure what enabling it costs."""
    return RouteOutcome(chunks=[], diagnostics={
        "note": "no-retrieval route: empty context; grounded generator will abstain",
    })


# --------------------------------------------------------------------------------------------
# Route B — single-step retrieval
# --------------------------------------------------------------------------------------------

def single_step(question: str, index: CorpusIndex, config: AdaptiveConfig,
                tracer: Tracer | None = None) -> RouteOutcome:
    """The paper's B route: one dense top-k pass, nothing else.

    Kept intentionally identical to the naive baseline so the benchmark delta between adaptive
    and naive isolates the value of *routing*, not of a beefed-up easy path."""
    trace = tracer or default_tracer
    with trace.span("adaptive.single_step", k=config.single_k) as span:
        started = time.perf_counter()
        hits = index.dense_search(question, config.single_k)
        latency_ms = (time.perf_counter() - started) * 1000.0
        span.set("hits", len(hits))
    return RouteOutcome(chunks=hits, diagnostics={
        "scores": [(h.chunk_id, round(h.score, 4)) for h in hits],
        "latency_ms": round(latency_ms, 2),
    })


# --------------------------------------------------------------------------------------------
# Route C — multi-step iterative retrieval
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FollowUpDecision:
    """One iteration's structured verdict: evidence is sufficient (``done``) or the named
    ``next_query`` should be retrieved next."""

    done: bool
    next_query: str
    reason: str


def _parse_follow_up(value: Any) -> FollowUpDecision:
    """Validator for ``StructuredCaller``: raises on bad shape, returns a typed decision."""
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object, got {type(value).__name__}")
    done = value["done"]
    if not isinstance(done, bool):
        raise TypeError(f"'done' must be a boolean, got {type(done).__name__}")
    next_query = str(value.get("next_query", "") or "").strip()
    reason = str(value.get("reason", "")).strip()
    if not done and not next_query:
        raise ValueError("'next_query' must be non-empty when done is false")
    return FollowUpDecision(done=done, next_query=next_query, reason=reason)


class MultiStepRetriever:
    """The paper's C route: an iterative retrieve-and-read evidence chain.

    Loop shape (see ARCHITECTURE.md for the diagram):

    1. **Seed** — fused (dense + BM25, RRF) retrieval on the original question starts the
       evidence pool; multi-hop questions still contain the first hop's entities verbatim.
    2. **Decide** — show the LLM the question plus accumulated evidence and ask whether the
       evidence suffices, or what additional information to retrieve next (structured output).
    3. **Retrieve** — run the follow-up query through the same fused search and append only
       chunks not already in evidence.
    4. Repeat from 2 until ``done``, evidence stalls, or ``max_iterations`` is reached.

    **Stall detection** (the loop's safety net): if a follow-up retrieval contributes zero new
    chunks — typically because the LLM keeps rephrasing the same gap and the index keeps
    returning the same hits — iterating further can only burn LLM calls on an unchanged
    evidence pool, since the next decision would see exactly the same input and (at
    temperature 0) reach exactly the same conclusion. We stop immediately and record
    ``stop_reason="stalled"``, deliberately preferring a truncated chain over a livelock.
    ``done``-with-unusable-output is likewise absorbed: if the decision call fails structured
    validation even after repair, we stop with ``stop_reason="decision_error"`` and answer from
    the evidence gathered so far — a degraded answer beats a crashed pipeline.

    Evidence keeps accumulation order (seed hits, then each hop's finds) rather than re-sorting
    by score: RRF scores from different iterations are not comparable, and the seed hits are the
    ones most directly aligned with the question.
    """

    def __init__(self, index: CorpusIndex, caller: StructuredCaller, config: AdaptiveConfig,
                 tracer: Tracer | None = None) -> None:
        self._index = index
        self._caller = caller
        self._config = config
        self._tracer = tracer or default_tracer

    # ---- public --------------------------------------------------------------------------

    def retrieve(self, question: str) -> RouteOutcome:
        cfg = self._config
        evidence: list[ScoredChunk] = []
        seen: set[str] = set()
        sub_queries: list[str] = []
        iterations: list[dict[str, Any]] = []

        with self._tracer.span("adaptive.multi_step", max_iterations=cfg.max_iterations) as span:
            # 1. Seed the evidence pool from the original question.
            seed_hits = self._fused_search(question, cfg.per_iteration_k)
            self._append_new(seed_hits, evidence, seen)
            seed_diag = {"query": question,
                         "chunk_ids": [h.chunk_id for h in seed_hits],
                         "doc_ids": _unique_doc_ids(seed_hits)}
            if not evidence:
                span.set("stop_reason", "no_seed_evidence")
                return RouteOutcome(chunks=[], diagnostics={
                    "seed": seed_diag, "iterations": [], "stop_reason": "no_seed_evidence"})

            # 2..n. Decide → retrieve follow-up → accumulate, until done/stall/budget.
            stop_reason = "max_iterations"
            for i in range(1, cfg.max_iterations + 1):
                with self._tracer.span("adaptive.iteration", n=i) as it_span:
                    record: dict[str, Any] = {"iteration": i}
                    try:
                        decision = self._decide(question, evidence)
                    except Exception as e:  # StructuredOutputError after repair retry
                        record.update(error=f"{type(e).__name__}: {e}")
                        iterations.append(record)
                        stop_reason = "decision_error"
                        break
                    record.update(done=decision.done, next_query=decision.next_query,
                                  reason=decision.reason, new_chunk_ids=[], new_doc_ids=[])
                    if decision.done:
                        iterations.append(record)
                        stop_reason = "classifier_done"
                        break

                    sub_queries.append(decision.next_query)
                    hits = self._fused_search(decision.next_query, cfg.per_iteration_k)
                    new = self._append_new(hits, evidence, seen)
                    record["new_chunk_ids"] = [h.chunk_id for h in new]
                    record["new_doc_ids"] = _unique_doc_ids(new)
                    it_span.set("new_chunks", len(new))
                    iterations.append(record)
                    if not new:
                        stop_reason = "stalled"
                        break
            span.set("iterations", len(iterations))
            span.set("stop_reason", stop_reason)
            span.set("evidence_chunks", len(evidence))

        return RouteOutcome(
            chunks=evidence,
            sub_queries=tuple(sub_queries),
            diagnostics={"seed": seed_diag, "iterations": iterations, "stop_reason": stop_reason},
        )

    # ---- internals -----------------------------------------------------------------------

    def _fused_search(self, query: str, k: int) -> list[ScoredChunk]:
        """Dense + BM25, fused with RRF, truncated to k. Hybrid on purpose: follow-up queries
        name specific entities learned from evidence ("Brightfen founder"), and exact names are
        BM25's home turf while dense covers the paraphrased ones."""
        dense = self._index.dense_search(query, k)
        sparse = self._index.sparse_search(query, k)
        return rrf([dense, sparse], k=self._config.rrf_k)[:k]

    def _decide(self, question: str, evidence: Sequence[ScoredChunk]) -> FollowUpDecision:
        prompt = FOLLOW_UP_PROMPT.format(question=question,
                                         evidence=self._format_evidence(evidence))
        return self._caller.call(prompt, validator=_parse_follow_up, max_tokens=300)

    def _format_evidence(self, evidence: Sequence[ScoredChunk]) -> str:
        """Numbered, per-passage-truncated evidence view for the decision prompt — the LLM needs
        each passage's gist, not its full text, and the prompt must not grow with hop count."""
        limit = self._config.evidence_max_chars_per_passage
        lines = []
        for n, hit in enumerate(evidence, start=1):
            text = hit.chunk.display_text.strip()
            if len(text) > limit:
                text = text[:limit].rstrip() + "…"
            lines.append(f"[{n}] {text}")
        return "\n".join(lines)

    @staticmethod
    def _append_new(hits: Sequence[ScoredChunk], evidence: list[ScoredChunk],
                    seen: set[str]) -> list[ScoredChunk]:
        """Append hits whose chunk_id is unseen; return just the newly added ones."""
        new: list[ScoredChunk] = []
        for hit in hits:
            if hit.chunk_id not in seen:
                seen.add(hit.chunk_id)
                evidence.append(hit)
                new.append(hit)
        return new


def _unique_doc_ids(hits: Sequence[ScoredChunk]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for h in hits:
        if h.doc_id not in seen:
            seen.add(h.doc_id)
            out.append(h.doc_id)
    return out
