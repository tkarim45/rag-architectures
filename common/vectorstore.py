"""Pluggable dense vector store — the swap seam the research calls for ("index builders write into
the vector store; swap stores without rewriting the app").

Default backend is **FAISS** (`IndexFlatIP`): because our embeddings are L2-normalized, inner
product == cosine, and a flat index is *exact* (no ANN approximation) — the right choice at this
corpus size, and a drop-in for an HNSW/IVF index when you scale up. A pure-NumPy backend is kept as
a zero-dependency fallback; both return identical results. Real deployments would add adapters for
Chroma / Qdrant / pgvector behind this same `add` / `search` interface.

Select with RAGARCH_VECTORSTORE=faiss|numpy (default faiss).
"""
from __future__ import annotations

import os

import numpy as np


class VectorStore:
    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...
    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]: ...


class NumpyStore(VectorStore):
    """Exact brute-force cosine over an in-memory matrix. Optimal at small scale."""

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.mat: np.ndarray | None = None

    def add(self, ids, vectors):
        self.ids = list(ids)
        self.mat = np.asarray(vectors, dtype=float)

    def search(self, query_vec, k):
        sims = self.mat @ np.asarray(query_vec, dtype=float)
        order = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in order]


class FaissStore(VectorStore):
    """FAISS IndexFlatIP — exact inner-product search (== cosine on normalized vectors)."""

    def __init__(self) -> None:
        import faiss
        self._faiss = faiss
        self.index = None
        self.ids: list[str] = []

    def add(self, ids, vectors):
        v = np.asarray(vectors, dtype="float32")
        self.ids = list(ids)
        self.index = self._faiss.IndexFlatIP(v.shape[1])
        self.index.add(v)

    def search(self, query_vec, k):
        q = np.asarray([query_vec], dtype="float32")
        scores, idx = self.index.search(q, min(k, len(self.ids)))
        return [(self.ids[i], float(scores[0][j])) for j, i in enumerate(idx[0]) if i >= 0]


def get_store(name: str | None = None) -> VectorStore:
    name = (name or os.getenv("RAGARCH_VECTORSTORE", "faiss")).lower()
    if name == "numpy":
        return NumpyStore()
    try:
        return FaissStore()
    except Exception:          # faiss not importable → exact NumPy fallback (same results)
        return NumpyStore()
