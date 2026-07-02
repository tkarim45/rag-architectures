"""Embedder abstraction. All embeddings in the framework are L2-normalized so cosine similarity
== inner product everywhere — vector stores exploit that invariant, so it is enforced here at the
boundary rather than trusted per-backend."""
from __future__ import annotations

import hashlib
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """(n, dim) float32, rows L2-normalized."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """(dim,) float32, L2-normalized."""
        ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class SentenceTransformerEmbedder:
    """Local sentence-transformers backend (default: all-MiniLM-L6-v2, 384-dim). Model loads
    lazily so importing the framework never pulls torch until an embedding is actually needed."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._dim: int | None = None

    def _model_(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._model_()
        return self._dim  # type: ignore[return-value]

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model_().encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True,
            show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


class HashingEmbedder:
    """Deterministic, dependency-free embedder for offline tests: token-hash bag-of-words into a
    fixed number of buckets, L2-normalized. Texts sharing vocabulary land near each other, which is
    enough signal for retrieval tests without any model download."""

    def __init__(self, dim: int = 256):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _bucket(self, token: str) -> int:
        return int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "big") % self._dim

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                out[i, self._bucket(token.strip(".,!?;:()[]\"'"))] += 1.0
        return l2_normalize(out)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]
