"""Entity extraction — the one LLM call GraphRAG makes per document at build time.

Runs offline (once, when the graph is built). Each document's text goes to Claude, which returns a
comma-separated list of the named things it mentions; those entities are what later stitch documents
together into the doc-doc graph.
"""
from __future__ import annotations

from common import providers

from .prompts import ENTITY_PROMPT


def extract_entities(text: str) -> set[str]:
    """Pull the named entities out of `text` as a normalized set (lowercase, stripped, no empties)."""
    raw = providers.complete(ENTITY_PROMPT.format(text=text), max_tokens=120)
    return {e.strip().lower() for e in raw.split(",") if e.strip()}
