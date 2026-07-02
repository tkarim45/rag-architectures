"""All LLM touchpoints for Adaptive-RAG. Exactly two, both structured:

* ``CLASSIFIER_PROMPT`` — the complexity classifier that *is* the architecture: one cheap call
  labels the question A/B/C so the executor can dispatch it to the cheapest sufficient route.
  Jeong et al. 2024 train a small seq2seq classifier for this; we implement it as a zero-shot
  LLM call with the same label semantics, which keeps the package self-contained.
* ``FOLLOW_UP_PROMPT`` — the multi-step route's per-iteration decision: given the evidence
  accumulated so far, either declare it sufficient or name the missing fact as the next
  retrieval query (the paper's iterative retriever–reader loop, in the spirit of IRCoT).

Both demand bare JSON; parsing, validation and the one-shot repair retry live in
``core.StructuredCaller`` — prompts here only have to be unambiguous.
"""
from __future__ import annotations

CLASSIFIER_PROMPT = """\
Classify the question complexity for a retrieval-augmented QA system over a document corpus.

Labels:
  A — general knowledge. The question is answerable from broad world knowledge alone; no corpus
      lookup is needed (e.g. definitions, famous facts, common-sense questions).
  B — single-step. The question is answerable from ONE document: a single retrieval pass will
      surface a passage that states the answer directly.
  C — multi-step. Answering requires CHAINING facts across documents: an intermediate entity or
      fact must be looked up first before the document containing the final answer can even be
      searched for (e.g. "who founded the company that makes X?" — first resolve which company
      makes X, then look up its founder).

Question: {question}

Reply with ONLY a JSON object, no prose, no code fences:
{{"label": "A" | "B" | "C", "reason": "<one sentence justifying the label>"}}
"""

FOLLOW_UP_PROMPT = """\
You are driving an iterative retrieval loop for the question below. The passages under
"Accumulated evidence" are everything retrieved so far.

Question: {question}

Accumulated evidence:
{evidence}

Does the accumulated evidence already contain every fact needed to answer the question?
What additional information, if any, must be retrieved next? If a fact is missing, phrase
next_query as a short standalone search query for that specific missing fact (name the entities
you learned from the evidence — do not repeat the original question verbatim).

Reply with ONLY a JSON object, no prose, no code fences:
{{"done": true | false, "next_query": "<search query for the missing fact, or empty string if done>", "reason": "<one sentence>"}}
"""
