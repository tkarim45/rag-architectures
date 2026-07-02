"""Tunables for the Corrective RAG (CRAG) pipeline.

Every knob the architecture exposes lives here so experiments are a `dataclasses.replace` away
and nothing hides in module constants. Defaults follow the paper's shape (Yan et al. 2024,
arXiv:2401.15884): retrieve wider than you answer (`initial_k` > `final_k`), demand real
confidence before trusting a grade, and cast an even wider net (`fallback_k`) when the first
retrieval is judged bad.
"""
from __future__ import annotations

from dataclasses import dataclass

from core import ConfigurationError


@dataclass(frozen=True)
class Config:
    """Corrective RAG configuration.

    Retrieval widths
    ----------------
    initial_k:  passages fetched by the first dense retrieval — the set the evaluator grades.
    fallback_k: per-retriever depth of the broadened dense ∪ BM25 sweep used when the initial
                retrieval is judged INCORRECT/AMBIGUOUS (the closed-corpus stand-in for the
                paper's web search — see `retriever.py`). Wider than `initial_k` on purpose:
                the whole point of the fallback is to look where the first pass didn't.
    final_k:    passages that survive into refinement and the answer context.

    Evaluator thresholds (verdict aggregation, paper §3.2–3.3)
    ----------------------------------------------------------
    correct_confidence:   a passage counts as *confidently correct* only at/above this. The
                          query verdict is CORRECT iff at least one passage clears it.
    incorrect_confidence: a passage counts as *confidently incorrect* only at/above this. The
                          verdict is INCORRECT iff *every* passage is confidently incorrect.
                          Anything in between → AMBIGUOUS (the combine branch). Raising either
                          threshold pushes more queries into AMBIGUOUS — the safe, expensive
                          branch.

    Knowledge refinement (decompose-then-recompose, paper §3.4)
    -----------------------------------------------------------
    refine_strips:       master switch for sentence-strip filtering. Off → selected passages
                         pass through whole.
    refine_min_passages: skip strip-grading when fewer than this many passages were selected.
                         Cost control: strip-grading costs ~(passages × strips) LLM calls, and
                         with one or two passages the context is small enough that filtering
                         buys little.

    Fusion / context
    ----------------
    rrf_k:                Reciprocal Rank Fusion damping constant (60 is canonical).
    chunker:              chunking strategy for the lazily built index. `fixed` by default —
                          multi-sentence chunks give the refiner actual strips to filter
                          (a `sentence` index makes refinement a no-op: one strip per passage).
    max_context_passages / max_context_chars: ContextBuilder budget for the answer prompt.
    """

    # retrieval widths
    initial_k: int = 8
    fallback_k: int = 12
    final_k: int = 5

    # evaluator thresholds
    correct_confidence: float = 0.7
    incorrect_confidence: float = 0.7

    # knowledge refinement
    refine_strips: bool = True
    refine_min_passages: int = 2

    # fusion / context
    rrf_k: int = 60
    chunker: str = "fixed"
    max_context_passages: int = 5
    max_context_chars: int = 6000

    def __post_init__(self) -> None:
        if self.initial_k < 1 or self.fallback_k < 1 or self.final_k < 1:
            raise ConfigurationError("initial_k, fallback_k and final_k must all be >= 1")
        for name in ("correct_confidence", "incorrect_confidence"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigurationError(f"{name} must be in [0, 1], got {value}")
        if self.refine_min_passages < 0:
            raise ConfigurationError("refine_min_passages must be >= 0")
        if self.rrf_k < 1:
            raise ConfigurationError("rrf_k must be >= 1")
