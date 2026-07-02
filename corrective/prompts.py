"""Every LLM touchpoint of the corrective package, in one place.

Three prompts, one per CRAG component (Yan et al. 2024, arXiv:2401.15884):

  * GRADE_PASSAGE   — the retrieval evaluator (§3.2). Structured JSON out via StructuredCaller.
  * STRIP_RELEVANCE — knowledge-refinement strip filter (§3.4). Deliberately a plain YES/NO
    completion, not JSON: it runs once per sentence strip, so it must be the cheapest possible
    call and trivially parseable.
  * REWRITE_QUERY   — the query rewriter that precedes the fallback search (§3.3). Plain text.

Keeping prompts here (and only here) means prompt iteration never touches control flow, and the
offline FakeLLM can route on the stable marker phrases ("Grade the retrieved passage",
"Is this strip relevant", "Rewrite the question").
"""
from __future__ import annotations

GRADE_PASSAGE = """\
You are a retrieval evaluator for a question-answering system.

Grade the retrieved passage below on whether it is relevant to answering the question. Judge
only informational relevance — not writing quality, and not whether it answers the question
completely on its own.

Question: {question}

Retrieved passage:
{passage}

Grading scale:
- "correct":   the passage contains information that directly helps answer the question.
- "incorrect": the passage is irrelevant or misleading for this question.
- "ambiguous": the passage is topically related but you cannot tell whether it helps.

Reply with ONLY a JSON object, no prose:
{{"grade": "correct" | "incorrect" | "ambiguous", "confidence": <number between 0 and 1>}}
"confidence" is how certain you are of the grade."""


STRIP_RELEVANCE = """\
Question: {question}

Knowledge strip:
{strip}

Is this strip relevant to answering the question? Answer with exactly one word: YES or NO."""


REWRITE_QUERY = """\
A search for the question below returned only irrelevant passages, so the search will be
retried against a wider index.

Rewrite the question as a short, keyword-focused search query. Keep every distinctive name and
term from the question; drop filler words. Do not answer the question.

Question: {question}

Reply with ONLY the rewritten query on a single line — no quotes, no explanation."""
