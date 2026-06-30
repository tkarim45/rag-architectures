"""Prompts for the CRAG self-check loop: a per-passage relevance grader and a query rewriter that
turns a weak question into a cleaner search query for the corrective re-retrieval."""
from __future__ import annotations

GRADE_PROMPT = (
    "Question: {query}\n"
    "Passage: {passage}\n"
    "Is this passage relevant to answering the question? Reply YES or NO."
)

REWRITE_PROMPT = (
    "Rewrite this question to be a clearer search query for a knowledge base:\n{query}"
)
