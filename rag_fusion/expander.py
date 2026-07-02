"""LLM query expansion for RAG-Fusion.

Design decisions:

* Structured output goes through ``core.StructuredCaller`` with a validator — the generated
  queries feed *code* (a retrieval fan-out into rank fusion), so they must parse into
  ``list[str]`` or be repaired, never trusted as free-form text.
* The original question is always kept and always first. Broadening exists to add recall; it must
  never be able to displace the user's actual intent from the fused ranking.
* Two dedup layers: exact (casefolded string) inside the validator, then semantic — under RRF a
  near-duplicate query is *worse* than useless, because its near-identical ranking double-counts
  the same documents' votes and skews the fusion. A candidate whose embedding cosine against any
  already-kept query reaches ``dedup_threshold`` is dropped. Embeddings from ``core.embeddings``
  are L2-normalized, so cosine is a plain dot product.
* Expansion failure degrades to the original question alone (flagged on the result) instead of
  failing retrieval: the system should never answer worse because an optional recall
  optimization broke.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core import StructuredCaller, StructuredOutputError, Tracer
from core.embeddings import Embedder
from core.llm import LLM

from .prompts import EXPANSION_PROMPT


def validate_query_list(value: object) -> list[str]:
    """Validator for the expansion call: require a JSON array of strings, strip whitespace, drop
    empties and exact (casefolded) duplicates. Raises on wrong shape so ``StructuredCaller`` can
    run its repair loop."""
    if not isinstance(value, list):
        raise TypeError(f"expected a JSON array of strings, got {type(value).__name__}")
    queries: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"array items must be strings, got {type(item).__name__}")
        text = item.strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        queries.append(text)
    if not queries:
        raise ValueError("array contained no usable query strings")
    return queries


@dataclass(frozen=True)
class Expansion:
    """What the expander decided to search, and why.

    ``queries`` is the searchable set (original question always at index 0); the other fields are
    the audit trail that ends up in ``RetrievalResult.diagnostics``.
    """

    queries: tuple[str, ...]
    llm_variants: tuple[str, ...] = ()   # what the LLM proposed, post-validation
    dropped: tuple[str, ...] = ()        # queries removed as near-duplicates of a kept one
    fallback: bool = False               # True when LLM output was unusable and we degraded


class QueryExpander:
    """Turns one question into a small, de-duplicated set of broadened search queries."""

    def __init__(self, llm: LLM, embedder: Embedder, tracer: Tracer, *, n_queries: int,
                 dedup_threshold: float, max_tokens: int = 300) -> None:
        self._caller = StructuredCaller(llm)
        self._embedder = embedder
        self._tracer = tracer
        self._n_queries = n_queries
        self._dedup_threshold = dedup_threshold
        self._max_tokens = max_tokens

    def expand(self, question: str) -> Expansion:
        with self._tracer.span("rag_fusion.expand", n_requested=self._n_queries) as span:
            try:
                variants = self._caller.call(
                    EXPANSION_PROMPT.format(n=self._n_queries, question=question),
                    validator=validate_query_list, max_tokens=self._max_tokens)
            except StructuredOutputError:
                span.set("fallback", True)
                return Expansion(queries=(question,), fallback=True)
            variants = variants[: self._n_queries]
            kept, dropped = self._semantic_dedup(question, variants)
            span.set("kept", len(kept))
            span.set("dropped", len(dropped))
        return Expansion(queries=tuple(kept), llm_variants=tuple(variants),
                         dropped=tuple(dropped))

    def _semantic_dedup(self, question: str,
                        variants: Sequence[str]) -> tuple[list[str], list[str]]:
        """Greedy filter in LLM order: keep a query only if its embedding stays below the cosine
        threshold against *every* query already kept (the original included)."""
        kept = [question]
        dropped = [v for v in variants if v.casefold() == question.casefold()]
        candidates = [v for v in variants if v.casefold() != question.casefold()]
        if not candidates:
            return kept, dropped
        vectors = self._embedder.embed_texts([question, *candidates])
        kept_vectors = [vectors[0]]
        for text, vector in zip(candidates, vectors[1:]):
            similarity = float(np.max(np.stack(kept_vectors) @ vector))  # normalized ⇒ dot=cosine
            if similarity >= self._dedup_threshold:
                dropped.append(text)
            else:
                kept.append(text)
                kept_vectors.append(vector)
        return kept, dropped
