"""All LLM touchpoints for the HyDE package.

One prompt: hypothesis generation. The phrase "hypothetical document" is load-bearing twice over —
it is the paper's term of art, and it is the routing substring offline tests use to target this
prompt with `FakeLLM.on("hypothetical document", ...)`.

The prompt deliberately *permits* invented facts. HyDE's insight is that the hypothesis is a
vocabulary/shape donor, not an answer: only its embedding neighborhood matters, and a confidently
wrong passage written in corpus register lands nearer the true passage than the question does.
"""
from __future__ import annotations

HYPOTHESIS_SYSTEM = (
    "You write short encyclopedia-style passages. Output only the passage itself - no preamble, "
    "no headings, no commentary."
)

HYPOTHESIS_PROMPT = (
    "Write a short hypothetical document (one paragraph, at most {max_words} words) that would "
    "directly answer the question below, in the style of a reference-corpus passage. Invent "
    "specific names, dates, and figures if you do not know them - the passage is used only for "
    "similarity search, never shown to a user, so plausible vocabulary matters more than factual "
    "accuracy.\n\nQuestion: {question}\n\nHypothetical document:"
)


def hypothesis_prompt(question: str, *, max_words: int = 120) -> str:
    """Render the hypothesis prompt for one question."""
    return HYPOTHESIS_PROMPT.format(question=question, max_words=max_words)
