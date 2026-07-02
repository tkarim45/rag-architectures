"""Layered runtime configuration.

Precedence: explicit constructor kwargs > environment variables > defaults. Environment loads from
the project `.env` and the global `~/.env` once, at import of this module — the same contract the
original repo had, kept because every consumer (CLI, benchmark, tests, notebooks) relies on it.

Architecture packages define their *own* config dataclasses (tunables local to that architecture);
this module only owns the cross-cutting knobs: which LLM backend, which model, which vector store,
cache locations, retry policy.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("USE_TF", "0")  # keep transformers off the Keras-3 TF path

try:  # pragma: no cover - dotenv presence is environmental
    from dotenv import load_dotenv

    load_dotenv()
    load_dotenv(os.path.expanduser("~/.env"))
except Exception:  # pragma: no cover
    pass

from .errors import ConfigurationError

_VALID_LLM_BACKENDS = ("bedrock", "anthropic")
_VALID_VECTORSTORES = ("faiss", "numpy")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from e


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ConfigurationError(f"{name} must be a float, got {raw!r}") from e


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter for provider calls."""

    max_attempts: int = 5
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    backoff_multiplier: float = 2.0


@dataclass(frozen=True)
class LLMConfig:
    backend: str = field(default_factory=lambda: _env("RAGARCH_LLM_BACKEND", "bedrock"))
    model: str = field(default_factory=lambda: _env(
        "RAGARCH_MODEL", "global.anthropic.claude-haiku-4-5-20251001-v1:0"))
    aws_region: str = field(default_factory=lambda: _env("AWS_REGION", "us-east-1"))
    temperature: float = field(default_factory=lambda: _env_float("RAGARCH_TEMPERATURE", 0.0))
    max_tokens_default: int = field(default_factory=lambda: _env_int("RAGARCH_MAX_TOKENS", 512))
    request_timeout_s: float = field(default_factory=lambda: _env_float("RAGARCH_LLM_TIMEOUT", 60.0))
    retry: RetryPolicy = field(default_factory=RetryPolicy)

    def __post_init__(self) -> None:
        if self.backend not in _VALID_LLM_BACKENDS:
            raise ConfigurationError(
                f"RAGARCH_LLM_BACKEND must be one of {_VALID_LLM_BACKENDS}, got {self.backend!r}")


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = field(default_factory=lambda: _env("RAGARCH_EMBED_MODEL", "all-MiniLM-L6-v2"))
    batch_size: int = field(default_factory=lambda: _env_int("RAGARCH_EMBED_BATCH", 64))


@dataclass(frozen=True)
class StoreConfig:
    vector_backend: str = field(default_factory=lambda: _env("RAGARCH_VECTORSTORE", "faiss"))

    def __post_init__(self) -> None:
        if self.vector_backend not in _VALID_VECTORSTORES:
            raise ConfigurationError(
                f"RAGARCH_VECTORSTORE must be one of {_VALID_VECTORSTORES}, "
                f"got {self.vector_backend!r}")


@dataclass(frozen=True)
class CacheConfig:
    """Disk caches make iterating on the benchmark affordable: repeated runs re-embed nothing and
    re-ask the LLM nothing for identical (model, prompt) pairs. Off by default for generation-time
    calls so measured runs stay honest; embedding cache is on (embeddings are deterministic)."""

    dir: Path = field(default_factory=lambda: Path(
        _env("RAGARCH_CACHE_DIR", os.path.expanduser("~/.cache/rag-architectures"))))
    embeddings: bool = field(default_factory=lambda: _env_bool("RAGARCH_CACHE_EMBEDDINGS", True))
    llm: bool = field(default_factory=lambda: _env_bool("RAGARCH_CACHE_LLM", False))


@dataclass(frozen=True)
class CoreConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    @classmethod
    def from_env(cls) -> "CoreConfig":
        return cls()
