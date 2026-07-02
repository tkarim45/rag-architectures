"""Lexical (sparse) index: BM25 behind the same hit-shaped interface as the vector stores.

The analyzer is explicit and configurable — lowercasing, token pattern, stopwords, and a light
suffix stemmer — because lexical retrieval quality is mostly analyzer quality, and hiding it
inside a library call is how BM25 gets unfairly benchmarked.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..errors import IndexError_
from .vector import VectorHit

_DEFAULT_STOPWORDS = frozenset(
    "a an and are as at be by for from has he in is it its of on or that the to was were will "
    "with what which who whom whose when where why how does did do done doing".split())

_SUFFIXES = ("ing", "edly", "ed", "es", "s", "ly")


@dataclass(frozen=True)
class Analyzer:
    """Text → terms. Deterministic and shared between index and query sides (mismatch between the
    two is the classic silent BM25 bug)."""

    token_pattern: str = r"[a-z0-9]+"
    stopwords: frozenset[str] = _DEFAULT_STOPWORDS
    stem: bool = True
    min_token_len: int = 2

    def __call__(self, text: str) -> list[str]:
        tokens = re.findall(self.token_pattern, text.lower())
        out: list[str] = []
        for token in tokens:
            if token in self.stopwords or len(token) < self.min_token_len:
                continue
            out.append(self._stem(token) if self.stem else token)
        return out

    @staticmethod
    def _stem(token: str) -> str:
        for suffix in _SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[: -len(suffix)]
        return token


@dataclass
class BM25Index:
    """Okapi BM25 with configurable k1/b, built on rank_bm25 for the scoring core but owning the
    analysis pipeline and the hit interface."""

    analyzer: Analyzer = field(default_factory=Analyzer)
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self._ids: list[str] = []
        self._bm25 = None

    def add(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        if self._bm25 is not None:
            raise IndexError_("BM25Index is build-once; create a new index to re-add")
        if len(ids) != len(texts):
            raise IndexError_(f"{len(ids)} ids but {len(texts)} texts")
        from rank_bm25 import BM25Okapi

        self._ids = list(ids)
        corpus_terms = [self.analyzer(t) or ["<empty>"] for t in texts]
        self._bm25 = BM25Okapi(corpus_terms, k1=self.k1, b=self.b)

    def search(self, query: str, k: int) -> list[VectorHit]:
        if self._bm25 is None:
            raise IndexError_("search before add — the index is empty")
        terms = self.analyzer(query)
        if not terms:
            return []
        scores = self._bm25.get_scores(terms)
        order = np.argsort(-scores)[:k]
        return [VectorHit(self._ids[i], float(scores[i])) for i in order if scores[i] > 0.0]

    def __len__(self) -> int:
        return len(self._ids)
