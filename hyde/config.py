"""Tunables for the HyDE pipeline.

Every knob that changes retrieval behaviour lives here, frozen, so a benchmark run can be
described completely by one `Config` value. Defaults follow Gao et al. 2022 in spirit but are
tuned for this repo's setup: a single hypothesis at temperature 0 (deterministic, cheapest), and
`query_weight=0.25` so the real query anchors the search vector against hallucinated-entity drift
(see ARCHITECTURE.md for the tradeoff).
"""
from __future__ import annotations

from dataclasses import dataclass

from core import ConfigurationError


@dataclass(frozen=True)
class Config:
    """HyDE tunables.

    Attributes:
        n_hypotheses: How many hypothetical documents to generate. With `temperature == 0` every
            call returns the same text, so values > 1 only add diversity (and recall) when
            `temperature > 0`. Default 1: one deterministic hypothesis.
        temperature: Sampling temperature for hypothesis generation. Keep 0.0 for reproducible
            benchmarks; raise (e.g. 0.7) when `n_hypotheses > 1` so the hypotheses actually differ.
        query_weight: Weight of the *real* query vector in the mixed search vector, in [0, 1].
            0.0 is paper-pure HyDE (search on hypotheses alone — maximal vocabulary transfer, but
            inherits any hallucinated entities); 1.0 degenerates to naive dense retrieval. The
            0.25 default keeps the hypothesis dominant while anchoring the search to what the user
            actually asked.
        hypothesis_max_tokens: Generation budget per hypothesis. Hypotheses only need to *look
            like* corpus passages; a short paragraph is enough and keeps the extra LLM call cheap.
        top_k: Chunks pulled from the dense index with the mixed vector.
        final_k: Passages handed to the generator (ContextBuilder `max_passages`).
        chunker: Which offline chunking strategy to index with
            (`whole | sentence | fixed | sentence_window | parent_child | contextual`).
        max_context_chars: Character budget for the assembled context block.
    """

    n_hypotheses: int = 1
    temperature: float = 0.0
    query_weight: float = 0.25
    hypothesis_max_tokens: int = 256
    top_k: int = 8
    final_k: int = 5
    chunker: str = "sentence"
    max_context_chars: int = 6000

    def __post_init__(self) -> None:
        if self.n_hypotheses < 1:
            raise ConfigurationError("n_hypotheses must be >= 1")
        if not 0.0 <= self.query_weight <= 1.0:
            raise ConfigurationError("query_weight must be in [0, 1]")
        if self.temperature < 0.0:
            raise ConfigurationError("temperature must be >= 0")
        if self.top_k < 1 or self.final_k < 1:
            raise ConfigurationError("top_k and final_k must be >= 1")
        if self.n_hypotheses > 1 and self.temperature == 0.0:
            # Not an error — but multiple identical hypotheses are wasted LLM calls, so make the
            # misconfiguration loud at construction time rather than silent at benchmark time.
            raise ConfigurationError(
                "n_hypotheses > 1 requires temperature > 0: at temperature 0 every hypothesis "
                "is identical, so the extra LLM calls buy nothing")
