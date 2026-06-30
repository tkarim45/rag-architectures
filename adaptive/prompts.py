"""Prompts for Adaptive RAG. The single LLM touchpoint is the router: one cheap classification call
that labels the query so it can be dispatched to the cheapest sufficient retriever."""
from __future__ import annotations

ROUTER_PROMPT = (
    "Classify the question into exactly one of: simple, multi_hop, broad.\n"
    "simple = one fact; multi_hop = needs chaining facts across entities; broad = overview/aggregation.\n"
    "Question: {query}\n"
    "Reply with only the label."
)
