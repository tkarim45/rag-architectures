"""The retrieval evaluator — the heart of CRAG (Yan et al. 2024, §3.2).

Standard RAG trusts whatever the retriever returns; CRAG's contribution is a lightweight
evaluator that grades EACH retrieved passage for relevance to the query and aggregates the
per-passage grades into a per-query verdict that drives the action policy:

    CORRECT    — at least one passage is confidently correct → trust internal knowledge, refine it.
    INCORRECT  — every passage is confidently incorrect      → discard retrieval, go to fallback.
    AMBIGUOUS  — anything in between                          → hedge: combine both sources.

The paper fine-tunes a T5 evaluator; here the grader is an LLM structured-output call, which
keeps the package model-agnostic and offline-testable. Grades flow through `StructuredCaller`
(parse → validate → one repair retry); a passage whose grade never parses degrades to
("ambiguous", 0.0) rather than raising — evaluator flakiness should push the pipeline toward
the cautious combine branch, not crash it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from core import ScoredChunk, StructuredCaller, StructuredOutputError, Tracer

from .config import Config
from .prompts import GRADE_PASSAGE

Grade = Literal["correct", "incorrect", "ambiguous"]
Verdict = Literal["correct", "incorrect", "ambiguous"]

_VALID_GRADES: frozenset[str] = frozenset({"correct", "incorrect", "ambiguous"})


@dataclass(frozen=True)
class PassageGrade:
    """One passage's relevance judgment, kept alongside its provenance for diagnostics."""

    chunk_id: str
    doc_id: str
    grade: Grade
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "doc_id": self.doc_id,
                "grade": self.grade, "confidence": self.confidence}


@dataclass(frozen=True)
class Evaluation:
    """All passage grades plus the aggregated per-query verdict."""

    grades: tuple[PassageGrade, ...]
    verdict: Verdict

    def grade_of(self, chunk_id: str) -> Grade:
        for g in self.grades:
            if g.chunk_id == chunk_id:
                return g.grade
        return "ambiguous"


def _validate_grade(value: Any) -> tuple[Grade, float]:
    """StructuredCaller validator: enforce the {"grade", "confidence"} shape and ranges."""
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object, got {type(value).__name__}")
    grade = str(value["grade"]).strip().lower()
    if grade not in _VALID_GRADES:
        raise ValueError(f"grade must be one of {sorted(_VALID_GRADES)}, got {grade!r}")
    confidence = float(value["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")
    return grade, confidence  # type: ignore[return-value]  # narrowed by the membership check


class RetrievalEvaluator:
    """Grades retrieved passages and aggregates them into the CRAG verdict."""

    def __init__(self, caller: StructuredCaller, config: Config, tracer: Tracer) -> None:
        self._caller = caller
        self._config = config
        self._tracer = tracer

    def evaluate(self, question: str, hits: Sequence[ScoredChunk]) -> Evaluation:
        """Grade every passage independently, then aggregate. One structured LLM call per
        passage — the cost CRAG pays for knowing when its retrieval is bad."""
        with self._tracer.span("corrective.evaluate", passages=len(hits)) as span:
            grades = tuple(self._grade_passage(question, hit) for hit in hits)
            verdict = self._aggregate(grades)
            span.set("verdict", verdict)
            span.set("grades", [g.grade for g in grades])
        return Evaluation(grades=grades, verdict=verdict)

    # ---- internals ---------------------------------------------------------------------

    def _grade_passage(self, question: str, hit: ScoredChunk) -> PassageGrade:
        prompt = GRADE_PASSAGE.format(question=question, passage=hit.chunk.display_text)
        try:
            grade, confidence = self._caller.call(prompt, validator=_validate_grade,
                                                  max_tokens=128)
        except StructuredOutputError:
            # Fail toward caution: an ungradeable passage is treated as ambiguous with zero
            # confidence, which can never certify CORRECT nor complete an INCORRECT sweep.
            grade, confidence = "ambiguous", 0.0
        return PassageGrade(chunk_id=hit.chunk_id, doc_id=hit.doc_id,
                            grade=grade, confidence=confidence)

    def _aggregate(self, grades: Sequence[PassageGrade]) -> Verdict:
        """Paper §3.3 aggregation: CORRECT if any passage is confidently correct; INCORRECT if
        all passages are confidently incorrect; AMBIGUOUS otherwise. An empty retrieval is
        INCORRECT by definition — there is nothing to trust, so go straight to fallback."""
        if not grades:
            return "incorrect"
        cfg = self._config
        if any(g.grade == "correct" and g.confidence >= cfg.correct_confidence for g in grades):
            return "correct"
        if all(g.grade == "incorrect" and g.confidence >= cfg.incorrect_confidence
               for g in grades):
            return "incorrect"
        return "ambiguous"
