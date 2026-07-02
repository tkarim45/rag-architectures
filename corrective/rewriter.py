"""Query rewriting for the corrective fallback (Yan et al. 2024, §3.3).

When the evaluator judges the initial retrieval INCORRECT (or AMBIGUOUS), CRAG does not re-run
the same query and hope — it rewrites the question into a keyword-style search query first, on
the theory that the *phrasing* of the question, not the corpus, may be what missed. The
rewritten query then drives the broadened fallback search in `retriever.py`.

Plain-text completion, defensively post-processed: models like to wrap rewrites in quotes or
add a second explanatory line, and a quoted query silently changes BM25 behavior. If the model
returns nothing usable, the original question is used — a failed rewrite must never turn into
an empty search.
"""
from __future__ import annotations

from core import LLM, Tracer

from .prompts import REWRITE_QUERY

_QUOTE_CHARS = "\"'`“”‘’"


class QueryRewriter:
    """Turns a natural-language question into a keyword-focused retry query."""

    def __init__(self, llm: LLM, tracer: Tracer) -> None:
        self._llm = llm
        self._tracer = tracer

    def rewrite(self, question: str) -> str:
        with self._tracer.span("corrective.rewrite") as span:
            raw = self._llm.complete_text(REWRITE_QUERY.format(question=question),
                                          max_tokens=64)
            rewritten = self._clean(raw) or question
            span.set("rewritten", rewritten)
        return rewritten

    @staticmethod
    def _clean(raw: str) -> str:
        """First line only, quotes stripped: everything past line one is model chatter."""
        first_line = raw.strip().splitlines()[0] if raw.strip() else ""
        return first_line.strip().strip(_QUOTE_CHARS).strip()
