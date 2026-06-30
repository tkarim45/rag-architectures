"""Prompts for GraphRAG. The single LLM touchpoint at build time is entity extraction — pulling the
named things out of each document so docs that mention the same entity can be linked into a graph."""
from __future__ import annotations

ENTITY_PROMPT = (
    "Extract the named entities (people, companies, products, languages, protocols) from this text "
    "as a comma-separated list, lowercase, no extra words:\n{text}"
)
