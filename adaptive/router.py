"""Query router — the one LLM call Adaptive RAG makes per query (online).

A lightweight classification: label the query simple / multi_hop / broad so the pipeline can dispatch
it to the cheapest sufficient retriever. Matching is substring-based and defaults to 'simple' so a
malformed or unexpected label degrades to the fast dense path rather than erroring.
"""
from __future__ import annotations

from common import providers

from .prompts import ROUTER_PROMPT


def route(query: str) -> str:
    """Return one of 'simple' | 'multi_hop' | 'broad' for `query`."""
    label = providers.complete(ROUTER_PROMPT.format(query=query), max_tokens=5).strip().lower()
    if "multi" in label:
        return "multi_hop"
    if "broad" in label:
        return "broad"
    return "simple"
