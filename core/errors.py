"""Error taxonomy. Architecture packages raise these instead of letting backend exceptions
(botocore, faiss, transformers) leak upward — callers can catch one family and decide policy
(retry / degrade / fail the benchmark row) without importing every backend's error types."""
from __future__ import annotations


class RagArchError(Exception):
    """Base class for all framework errors."""


class ConfigurationError(RagArchError):
    """Invalid or missing configuration (bad env var, unknown backend name, missing creds)."""


class ProviderError(RagArchError):
    """An external provider (LLM, embedder) failed after retries were exhausted."""

    def __init__(self, message: str, *, provider: str = "", retryable: bool = False):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class RateLimitError(ProviderError):
    """Provider throttled us. Always retryable; carried separately so backoff can be harsher."""

    def __init__(self, message: str, *, provider: str = ""):
        super().__init__(message, provider=provider, retryable=True)


class StructuredOutputError(ProviderError):
    """The LLM's output could not be parsed into the requested structure after all repair
    attempts. Carries the raw text so callers can log or fall back."""

    def __init__(self, message: str, *, raw: str = "", provider: str = ""):
        super().__init__(message, provider=provider, retryable=False)
        self.raw = raw


class IndexError_(RagArchError):
    """Index lifecycle misuse (searching before ingest, duplicate chunk ids, dim mismatch)."""


class RetrievalError(RagArchError):
    """A retrieval stage failed in a way the pipeline cannot degrade around."""
