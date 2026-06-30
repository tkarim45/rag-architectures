"""Relevance grader — the CRAG self-check. Asks the LLM whether a single retrieved passage actually
helps answer the query, so the pipeline can detect a bad first retrieval instead of answering from
it blindly."""
from __future__ import annotations

from common import providers

from .prompts import GRADE_PROMPT


def is_relevant(query: str, passage: str) -> bool:
    out = providers.complete(GRADE_PROMPT.format(query=query, passage=passage), max_tokens=5)
    return out.strip().upper().startswith("YES")
