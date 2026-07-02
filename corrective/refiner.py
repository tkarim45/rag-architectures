"""Knowledge refinement — CRAG's decompose-then-recompose step (Yan et al. 2024, §3.4).

A passage that is *relevant overall* still drags irrelevant sentences into the answer context,
and those sentences are exactly where grounded generators pick up distractors. The refiner:

  1. decomposes each selected passage into sentence-level "knowledge strips"
     (`core.ingestion.split_sentences` — the same splitter the chunkers use),
  2. grades each strip with the cheapest possible LLM call (plain YES/NO, no JSON),
  3. recomposes the kept strips *in original order* into a refined passage.

Provenance: refined output replaces the passage's display text, so we synthesize a new chunk
with `chunk_id = f"{original}::refined"` and the SAME `doc_id`. Chunk ids in this repo are
`"{doc_id}::spec"`, so the refined id still resolves to its document (citations split on
`"::"` and take the head) and `RetrievalResult.doc_ids` — what the benchmark scores — is
unchanged by refinement. The original chunk id is also kept in the refined chunk's metadata.

Cost control: strip-grading costs ~(passages × strips) LLM calls. It is skipped entirely when
`refine_strips` is off or when fewer than `refine_min_passages` passages were selected, and
single-strip passages are kept whole without a call (grading one strip is just re-grading the
passage the evaluator already graded).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from core import LLM, Chunk, ScoredChunk, Tracer
from core.ingestion import split_sentences

from .config import Config
from .prompts import STRIP_RELEVANCE

_REFINED_SUFFIX = "::refined"


@dataclass(frozen=True)
class RefinementReport:
    """What refinement did, for diagnostics: counts, not content."""

    applied: bool
    reason: str
    passages_in: int
    passages_out: int
    strips_kept: int
    strips_dropped: int

    def to_dict(self) -> dict[str, Any]:
        return {"applied": self.applied, "reason": self.reason,
                "passages_in": self.passages_in, "passages_out": self.passages_out,
                "strips_kept": self.strips_kept, "strips_dropped": self.strips_dropped}


def _refined_hit(hit: ScoredChunk, kept_strips: Sequence[str]) -> ScoredChunk:
    """Synthesize the refined chunk: same doc_id (provenance survives), derived chunk_id,
    display text recomposed from the kept strips."""
    original = hit.chunk
    refined = Chunk(
        chunk_id=f"{original.chunk_id}{_REFINED_SUFFIX}",
        doc_id=original.doc_id,
        index_text=original.index_text,
        display_text=" ".join(kept_strips),
        metadata=dict(original.metadata, refined=True, source_chunk=original.chunk_id),
    )
    return ScoredChunk(chunk=refined, score=hit.score, retriever="corrective.refined")


class KnowledgeRefiner:
    """Filters selected passages down to their question-relevant sentences."""

    def __init__(self, llm: LLM, config: Config, tracer: Tracer) -> None:
        self._llm = llm
        self._config = config
        self._tracer = tracer

    def refine(self, question: str,
               hits: Sequence[ScoredChunk]) -> tuple[list[ScoredChunk], RefinementReport]:
        """Decompose-then-recompose over `hits`. Returns the refined ranking plus a report.

        Safety valve: if strip-grading drops *everything* (a miscalibrated strip grader), the
        original passages are returned unrefined — an over-aggressive filter must degrade to
        plain RAG, not to an empty context."""
        cfg = self._config
        if not cfg.refine_strips:
            return list(hits), self._skipped(hits, "refine_strips disabled")
        if len(hits) < cfg.refine_min_passages:
            return list(hits), self._skipped(
                hits, f"passage count {len(hits)} < refine_min_passages {cfg.refine_min_passages}")

        with self._tracer.span("corrective.refine", passages=len(hits)) as span:
            refined: list[ScoredChunk] = []
            kept_total = 0
            dropped_total = 0
            for hit in hits:
                strips = split_sentences(hit.chunk.display_text)
                if len(strips) <= 1:
                    # Nothing to decompose — keep whole, spend no LLM call.
                    refined.append(hit)
                    kept_total += len(strips)
                    continue
                kept = [s for s in strips if self._strip_is_relevant(question, s)]
                kept_total += len(kept)
                dropped_total += len(strips) - len(kept)
                if kept:
                    refined.append(_refined_hit(hit, kept))
                # A passage with zero relevant strips is dropped outright.

            if not refined:
                span.set("outcome", "all_strips_dropped")
                report = RefinementReport(
                    applied=False, reason="all strips dropped — fell back to unrefined passages",
                    passages_in=len(hits), passages_out=len(hits),
                    strips_kept=0, strips_dropped=dropped_total)
                return list(hits), report

            span.set("strips_kept", kept_total)
            span.set("strips_dropped", dropped_total)
            report = RefinementReport(
                applied=True, reason="decompose-then-recompose",
                passages_in=len(hits), passages_out=len(refined),
                strips_kept=kept_total, strips_dropped=dropped_total)
            return refined, report

    # ---- internals ---------------------------------------------------------------------

    def _strip_is_relevant(self, question: str, strip: str) -> bool:
        """One cheap YES/NO completion per strip. Anything that doesn't start with YES is a
        drop — the conservative reading of a noisy grader."""
        reply = self._llm.complete_text(
            STRIP_RELEVANCE.format(question=question, strip=strip), max_tokens=8)
        return reply.strip().upper().startswith("YES")

    @staticmethod
    def _skipped(hits: Sequence[ScoredChunk], reason: str) -> RefinementReport:
        return RefinementReport(applied=False, reason=reason,
                                passages_in=len(hits), passages_out=len(hits),
                                strips_kept=0, strips_dropped=0)
