"""Draft a hypothetical answer for a query — the document HyDE embeds instead of the question."""
from __future__ import annotations

from common import providers

from .prompts import HYDE_PROMPT


def generate_hypothesis(query: str, max_tokens: int = 160) -> str:
    return providers.complete(HYDE_PROMPT.format(query=query), max_tokens)
