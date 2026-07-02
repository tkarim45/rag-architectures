"""Rerankers — stage 2 of the funnel: re-score (query, chunk) pairs with a sharper model.

Why a Protocol: the pipeline only needs "given a query and candidates, return them re-scored and
re-sorted". Structural typing keeps the heavy dependency (sentence-transformers) out of the
package's import graph — offline tests and the benchmark inject ``LexicalOverlapReranker`` and
never touch torch.

Why a cross-encoder beats the stage-1 bi-encoder: a bi-encoder embeds query and passage
*independently* and compares single vectors — cheap (passage vectors precomputed offline) but
coarse, since all interaction is squeezed through one dot product. A cross-encoder feeds the
*concatenated* query+passage through full transformer attention, so every query token attends to
every passage token — far sharper relevance judgments, but O(candidates) forward passes per query
with nothing precomputable. Hence the funnel: bi-encoder for recall over the whole corpus,
cross-encoder for precision over ``candidate_k`` survivors (Nogueira & Cho 2019).
"""
from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from core import ConfigurationError, ScoredChunk

_TOKEN = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Reranker(Protocol):
    """Anything that can re-score and re-sort candidates against a query."""

    name: str

    def rerank(self, query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        """Return `chunks` re-scored against `query`, sorted best-first. Must not add or invent
        chunks — stage 2 only reorders what stage 1 surfaced."""
        ...


class CrossEncoderReranker:
    """MS MARCO-style cross-encoder reranker over ``sentence_transformers.CrossEncoder``.

    The model is **lazy-loaded** on first use: constructing the pipeline (and importing this
    package) never downloads weights or imports torch, which keeps offline tests and
    `--help`-style CLI paths instant. Scores (query, ``chunk.index_text``) pairs — the same text
    the indexes matched on, so stage 1 and stage 2 judge the same evidence — in ``batch_size``
    groups to bound peak memory.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", *,
                 batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.name = f"cross_encoder:{model_name}"
        self._model = None  # loaded on first rerank()

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - environment-dependent
                raise ConfigurationError(
                    "CrossEncoderReranker requires sentence-transformers "
                    "(pip install sentence-transformers), or inject a Reranker such as "
                    "rerank.LexicalOverlapReranker") from exc
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        if not chunks:
            return []
        model = self._load()
        pairs = [(query, hit.chunk.index_text) for hit in chunks]
        scores = model.predict(pairs, batch_size=self.batch_size)
        rescored = [ScoredChunk(hit.chunk, float(score), retriever="rerank.cross_encoder")
                    for hit, score in zip(chunks, scores)]
        # sorted() is stable: ties keep stage-1 order, so results are deterministic.
        return sorted(rescored, key=lambda s: -s.score)


class LexicalOverlapReranker:
    """Deterministic, dependency-free reranker: token-set Jaccard between query and chunk.

    Two jobs by design:
    * **Offline tests** — the benchmark and CI inject it so the two-stage *plumbing* (candidate
      union, threshold, diagnostics, rank-movement accounting) is exercised without torch,
      network, or model downloads, and with reproducible scores.
    * **Documented fallback** — when the cross-encoder cannot load (air-gapped box, missing
      dependency), lexical overlap is a transparent, explainable stand-in. It is *weaker* than a
      cross-encoder (no synonymy, no word order) — a fallback, not a substitute.
    """

    name = "lexical_overlap"

    @staticmethod
    def _tokens(text: str) -> frozenset[str]:
        return frozenset(_TOKEN.findall(text.lower()))

    def rerank(self, query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        query_tokens = self._tokens(query)
        rescored: list[ScoredChunk] = []
        for hit in chunks:
            chunk_tokens = self._tokens(hit.chunk.index_text)
            union = query_tokens | chunk_tokens
            score = len(query_tokens & chunk_tokens) / len(union) if union else 0.0
            rescored.append(ScoredChunk(hit.chunk, score, retriever="rerank.lexical_overlap"))
        # sorted() is stable: ties keep stage-1 order, so results are reproducible.
        return sorted(rescored, key=lambda s: -s.score)
