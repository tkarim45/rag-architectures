"""Disk-backed embedding cache (SQLite, one row per (model, sha256(text)) pair).

Embeddings are deterministic, so caching is always safe — and it is the difference between a
benchmark iteration that re-encodes the whole corpus and one that reads it back in milliseconds.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Sequence

import numpy as np

from .base import Embedder

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    model TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (model, text_hash)
);
"""


class CachingEmbedder:
    def __init__(self, inner: Embedder, cache_dir: Path, *, model_name: str = ""):
        self.inner = inner
        self.model_name = model_name or getattr(inner, "model_name", type(inner).__name__)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(cache_dir / "embeddings.sqlite"), check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def dim(self) -> int:
        return self.inner.dim

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return self.inner.embed_texts(texts)
        hashes = [self._hash(t) for t in texts]
        found: dict[int, np.ndarray] = {}
        with self._lock:
            for i, h in enumerate(hashes):
                row = self._conn.execute(
                    "SELECT dim, vector FROM embeddings WHERE model = ? AND text_hash = ?",
                    (self.model_name, h)).fetchone()
                if row is not None:
                    found[i] = np.frombuffer(row[1], dtype=np.float32).reshape(row[0])
        missing = [i for i in range(len(texts)) if i not in found]
        self.hits += len(found)
        self.misses += len(missing)
        if missing:
            fresh = self.inner.embed_texts([texts[i] for i in missing])
            with self._lock:
                for j, i in enumerate(missing):
                    found[i] = fresh[j]
                    self._conn.execute(
                        "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)",
                        (self.model_name, hashes[i], fresh.shape[1], fresh[j].tobytes()))
                self._conn.commit()
        return np.stack([found[i] for i in range(len(texts))])

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]
