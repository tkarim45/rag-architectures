"""Prompt for drafting the hypothetical answer that HyDE embeds in place of the raw query."""
from __future__ import annotations

HYDE_PROMPT = (
    "Write a short factual passage (2-3 sentences) that would directly answer this "
    "question, even if you must invent plausible specifics:\n{query}"
)
