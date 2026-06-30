"""Chunking strategies + the dense/BM25 indexes they feed.

A Chunk separates `index_text` (what gets embedded / keyword-searched) from `return_text` (what is
handed to the generator) — that split is exactly what sentence-window and parent-child exploit:
match on something small and precise, return something larger and richer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from . import providers
from .corpus import Doc
from .vectorstore import get_store


@dataclass
class Chunk:
    cid: str
    doc_id: str
    index_text: str     # embedded + BM25-indexed
    return_text: str    # passed to the generator on a hit


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---- chunkers: each maps docs -> chunks ----
def chunk_whole(docs: list[Doc]) -> list[Chunk]:
    return [Chunk(d.id, d.id, d.text, d.text) for d in docs]


def chunk_sentence(docs: list[Doc]) -> list[Chunk]:
    out = []
    for d in docs:
        for i, s in enumerate(_sentences(d.text)):
            out.append(Chunk(f"{d.id}_s{i}", d.id, s, s))
    return out


def chunk_sentence_window(docs: list[Doc], window: int = 1) -> list[Chunk]:
    out = []
    for d in docs:
        sents = _sentences(d.text)
        for i, s in enumerate(sents):
            lo, hi = max(0, i - window), min(len(sents), i + window + 1)
            out.append(Chunk(f"{d.id}_w{i}", d.id, s, " ".join(sents[lo:hi])))
    return out


def chunk_parent_child(docs: list[Doc]) -> list[Chunk]:
    # index sentences (precise), return the whole doc (context)
    out = []
    for d in docs:
        for i, s in enumerate(_sentences(d.text)):
            out.append(Chunk(f"{d.id}_p{i}", d.id, s, d.text))
    return out


def chunk_contextual(docs: list[Doc]) -> list[Chunk]:
    # Anthropic-style contextual retrieval: prepend an LLM one-liner of doc context to each chunk
    out = []
    for d in docs:
        ctx = providers.complete(
            f"Document:\n{d.text}\n\nIn one short sentence, give the context a reader needs to "
            f"situate any excerpt from this document (who/what it is about). Reply with only the sentence.",
            max_tokens=60)
        for i, s in enumerate(_sentences(d.text)):
            out.append(Chunk(f"{d.id}_c{i}", d.id, f"{ctx} {s}", s))
    return out


CHUNKERS = {
    "whole": chunk_whole,
    "sentence": chunk_sentence,
    "sentence_window": chunk_sentence_window,
    "parent_child": chunk_parent_child,
    "contextual": chunk_contextual,
}


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


class Index:
    """Holds chunks + a dense vector store (FAISS by default) + a BM25 index over the same chunks.
    The dense half is backed by a pluggable VectorStore (see common/vectorstore.py); the sparse half
    is BM25. Embedding (offline) happens once here; retrieval (online) goes through the store."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.by_cid = {c.cid: c for c in chunks}
        embeddings = providers.embed([c.index_text for c in chunks])
        self.store = get_store()                       # FAISS IndexFlatIP (exact) by default
        self.store.add([c.cid for c in chunks], embeddings)
        self._bm25 = BM25Okapi([_tok(c.index_text) for c in chunks])

    def dense(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        return self.store.search(query_vec, k)

    def sparse(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(_tok(query))
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i].cid, float(scores[i])) for i in order]

    def doc_of(self, cid: str) -> str:
        return self.by_cid[cid].doc_id

    def text_of(self, cid: str) -> str:
        return self.by_cid[cid].return_text
