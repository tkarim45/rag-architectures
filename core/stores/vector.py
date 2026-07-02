"""Pluggable dense vector stores.

Contract: vectors arriving here are already L2-normalized (core.embeddings enforces it), so inner
product == cosine and every backend returns identical rankings. `FaissFlatStore` (exact,
IndexFlatIP) is the default; `NumpyStore` is the zero-dependency fallback. Both support metadata
filters and disk persistence — the same `add`/`search`/`save`/`load` seam a hosted store
(Qdrant / pgvector / OpenSearch) would slot into.

At this corpus size a flat exact index is optimal; ANN (HNSW/IVF) is a scale decision, not an
architecture decision, which is exactly why it hides behind this interface.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ..errors import IndexError_

MetadataFilter = Callable[[Mapping[str, Any]], bool]


@dataclass(frozen=True)
class VectorHit:
    id: str
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    def add(self, ids: Sequence[str], vectors: np.ndarray,
            metadata: Sequence[Mapping[str, Any]] | None = None) -> None: ...

    def search(self, query_vector: np.ndarray, k: int,
               where: MetadataFilter | None = None) -> list[VectorHit]: ...

    def __len__(self) -> int: ...


class NumpyStore:
    """Exact brute-force inner product over an in-memory matrix."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._meta: list[Mapping[str, Any]] = []
        self._matrix: np.ndarray | None = None

    def add(self, ids, vectors, metadata=None):
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(ids) != vectors.shape[0]:
            raise IndexError_(f"{len(ids)} ids but {vectors.shape[0]} vectors")
        dupes = set(ids) & set(self._ids)
        if dupes:
            raise IndexError_(f"duplicate vector ids: {sorted(dupes)[:5]}")
        metadata = list(metadata) if metadata is not None else [{} for _ in ids]
        self._ids.extend(ids)
        self._meta.extend(metadata)
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])

    def search(self, query_vector, k, where=None):
        if self._matrix is None:
            raise IndexError_("search before add — the store is empty")
        scores = self._matrix @ np.asarray(query_vector, dtype=np.float32)
        order = np.argsort(-scores)
        hits: list[VectorHit] = []
        for i in order:
            if where is not None and not where(self._meta[i]):
                continue
            hits.append(VectorHit(self._ids[i], float(scores[i]), self._meta[i]))
            if len(hits) >= k:
                break
        return hits

    def __len__(self) -> int:
        return len(self._ids)

    # ---- persistence -------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self._matrix if self._matrix is not None
                else np.zeros((0, 0), dtype=np.float32))
        (path / "ids.json").write_text(json.dumps({"ids": self._ids, "meta": self._meta}))

    @classmethod
    def load(cls, path: Path) -> "NumpyStore":
        store = cls()
        payload = json.loads((path / "ids.json").read_text())
        matrix = np.load(path / "vectors.npy")
        store._ids = payload["ids"]
        store._meta = payload["meta"]
        store._matrix = matrix if matrix.size else None
        return store


class FaissFlatStore:
    """FAISS IndexFlatIP — exact inner-product search. Metadata + ids live alongside the index
    (FAISS itself only knows row numbers)."""

    def __init__(self) -> None:
        import faiss

        self._faiss = faiss
        self._index = None
        self._ids: list[str] = []
        self._meta: list[Mapping[str, Any]] = []

    def add(self, ids, vectors, metadata=None):
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(ids) != vectors.shape[0]:
            raise IndexError_(f"{len(ids)} ids but {vectors.shape[0]} vectors")
        dupes = set(ids) & set(self._ids)
        if dupes:
            raise IndexError_(f"duplicate vector ids: {sorted(dupes)[:5]}")
        if self._index is None:
            self._index = self._faiss.IndexFlatIP(vectors.shape[1])
        elif vectors.shape[1] != self._index.d:
            raise IndexError_(f"dim mismatch: index is {self._index.d}, got {vectors.shape[1]}")
        self._index.add(vectors)
        self._ids.extend(ids)
        self._meta.extend(list(metadata) if metadata is not None else [{} for _ in ids])

    def search(self, query_vector, k, where=None):
        if self._index is None:
            raise IndexError_("search before add — the store is empty")
        # over-fetch when filtering: post-filter on metadata needs candidates beyond k
        fetch = min(len(self._ids), k if where is None else max(k * 4, k + 16))
        query = np.asarray([query_vector], dtype=np.float32)
        scores, rows = self._index.search(query, fetch)
        hits: list[VectorHit] = []
        for score, row in zip(scores[0], rows[0]):
            if row < 0:
                continue
            if where is not None and not where(self._meta[row]):
                continue
            hits.append(VectorHit(self._ids[row], float(score), self._meta[row]))
            if len(hits) >= k:
                break
        return hits

    def __len__(self) -> int:
        return len(self._ids)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            self._faiss.write_index(self._index, str(path / "index.faiss"))
        (path / "ids.json").write_text(json.dumps({"ids": self._ids, "meta": self._meta}))

    @classmethod
    def load(cls, path: Path) -> "FaissFlatStore":
        store = cls()
        payload = json.loads((path / "ids.json").read_text())
        store._ids = payload["ids"]
        store._meta = payload["meta"]
        index_file = path / "index.faiss"
        if index_file.exists():
            store._index = store._faiss.read_index(str(index_file))
        return store


def build_vector_store(backend: str = "faiss") -> VectorStore:
    if backend == "numpy":
        return NumpyStore()
    try:
        return FaissFlatStore()
    except ImportError:
        return NumpyStore()  # identical results, no faiss dependency
