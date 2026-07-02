"""Disk-backed LLM completion cache (SQLite).

Keyed on SHA-256 of (model, system, prompt, max_tokens, temperature, stop) — i.e. the full request
identity. Only sensible for deterministic (temperature 0) workloads such as offline indexing
(entity extraction, contextual chunk prefixes, RAPTOR summaries); generation-time caching is off by
default in `CacheConfig` so measured benchmark runs stay honest.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from ..types import TokenUsage
from .base import LLM, Completion, CompletionRequest

_SCHEMA = """
CREATE TABLE IF NOT EXISTS completions (
    key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    text TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    stop_reason TEXT NOT NULL DEFAULT ''
);
"""


class CachingLLM:
    """Decorator over any `LLM`. Thread-safe; one connection guarded by a lock is plenty at the
    call rates an LLM permits."""

    def __init__(self, inner: LLM, cache_dir: Path, *, model_hint: str = ""):
        self.inner = inner
        self.model_hint = model_hint or getattr(getattr(inner, "config", None), "model", "")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(cache_dir / "llm_cache.sqlite"), check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _key(self, request: CompletionRequest) -> str:
        identity = json.dumps({
            "model": self.model_hint,
            "system": request.system,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stop": list(request.stop_sequences),
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def complete(self, request: CompletionRequest) -> Completion:
        if request.temperature != 0.0:            # non-deterministic — never cache
            return self.inner.complete(request)
        key = self._key(request)
        with self._lock:
            row = self._conn.execute(
                "SELECT text, input_tokens, output_tokens, stop_reason FROM completions "
                "WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self.hits += 1
            return Completion(text=row[0], usage=TokenUsage(row[1], row[2]),
                              model=self.model_hint, stop_reason=row[3])
        self.misses += 1
        completion = self.inner.complete(request)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO completions VALUES (?, ?, ?, ?, ?, ?)",
                (key, completion.model, completion.text,
                 completion.usage.input_tokens, completion.usage.output_tokens,
                 completion.stop_reason))
            self._conn.commit()
        return completion

    def complete_text(self, prompt: str, *, system: str | None = None, max_tokens: int = 512,
                      temperature: float = 0.0, stop_sequences=()) -> str:
        return self.complete(CompletionRequest(
            prompt=prompt, system=system, max_tokens=max_tokens, temperature=temperature,
            stop_sequences=tuple(stop_sequences))).text.strip()
