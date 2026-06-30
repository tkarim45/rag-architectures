"""Corrective action — when too few retrieved chunks pass the relevance bar, rewrite the query into
a cleaner search query before re-retrieving (the closed-corpus stand-in for CRAG's web-search
fallback)."""
from __future__ import annotations

from common import providers

from .prompts import REWRITE_PROMPT


def rewrite(query: str) -> str:
    return providers.complete(REWRITE_PROMPT.format(query=query), max_tokens=60).strip()
