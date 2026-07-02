"""All LLM prompt text for the RAG-Fusion architecture lives here.

Keeping prompts out of the control flow makes them diffable, reviewable, and testable: offline
tests route on the distinctive phrase "broaden and diversify" with ``FakeLLM.on(...)``, so the
wording below is part of the package's contract with its test suite.

Unlike multi_query's rephrasing prompt, this one asks for queries that attack the information
need from *different angles* — RRF then rewards documents that survive across those angles.
"""
from __future__ import annotations

EXPANSION_PROMPT = """\
You generate search queries for a retrieval system that fuses ranked results across queries. \
Given the question below, write {n} search queries that broaden and diversify it: approach the \
same information need from different angles — sub-aspects, related entities, background context, \
alternate vocabulary — so that fusing the per-query rankings surfaces documents any single query \
would miss.

Rules:
- Every query must still serve the original question's intent; no topic drift.
- Vary the angle of attack across queries, not merely the wording.
- Reply with ONLY a JSON array of {n} strings. No prose, no code fences.

Question: {question}
"""
