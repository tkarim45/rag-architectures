"""LLM client abstraction.

Architecture packages depend on the `LLM` protocol only — never on a concrete SDK. That is what
makes every package testable offline (inject `FakeLLM`) and lets one env var swap Bedrock for the
direct Anthropic API without touching pipeline code.
"""
from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from ..config import RetryPolicy
from ..errors import ProviderError, RateLimitError
from ..telemetry import logger
from ..types import TokenUsage


@dataclass(frozen=True)
class CompletionRequest:
    prompt: str
    system: str | None = None
    max_tokens: int = 512
    temperature: float = 0.0
    stop_sequences: tuple[str, ...] = ()


@dataclass(frozen=True)
class Completion:
    text: str
    usage: TokenUsage = TokenUsage()
    model: str = ""
    latency_ms: float = 0.0
    stop_reason: str = ""


@runtime_checkable
class LLM(Protocol):
    """Minimal surface every backend implements."""

    def complete(self, request: CompletionRequest) -> Completion: ...


class BaseLLM(ABC):
    """Shared retry/accounting machinery. Subclasses implement one raw call; this class turns it
    into a resilient, observable client: exponential backoff with full jitter, rate-limit-aware
    delays, cumulative token accounting."""

    def __init__(self, retry: RetryPolicy | None = None):
        self.retry = retry or RetryPolicy()
        self.total_usage = TokenUsage()
        self.call_count = 0

    @abstractmethod
    def _invoke(self, request: CompletionRequest) -> Completion:
        """One raw provider call. Raise RateLimitError / ProviderError(retryable=...) on failure."""

    def complete(self, request: CompletionRequest) -> Completion:
        last_error: ProviderError | None = None
        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                started = time.perf_counter()
                completion = self._invoke(request)
                latency = (time.perf_counter() - started) * 1000.0
                completion = Completion(
                    text=completion.text, usage=completion.usage, model=completion.model,
                    latency_ms=latency, stop_reason=completion.stop_reason)
                self.call_count += 1
                self.total_usage = self.total_usage + completion.usage
                return completion
            except ProviderError as e:
                last_error = e
                if not e.retryable or attempt == self.retry.max_attempts:
                    raise
                delay = min(self.retry.max_delay_s,
                            self.retry.base_delay_s * self.retry.backoff_multiplier ** (attempt - 1))
                if isinstance(e, RateLimitError):
                    delay = min(self.retry.max_delay_s, delay * 2)
                delay *= random.uniform(0.5, 1.0)          # full jitter, decorrelates clients
                logger.warning("LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                               attempt, self.retry.max_attempts, e, delay)
                time.sleep(delay)
        raise last_error or ProviderError("LLM call failed with no recorded error")

    # ---- convenience -------------------------------------------------------------------

    def complete_text(self, prompt: str, *, system: str | None = None, max_tokens: int = 512,
                      temperature: float = 0.0, stop_sequences: Sequence[str] = ()) -> str:
        return self.complete(CompletionRequest(
            prompt=prompt, system=system, max_tokens=max_tokens, temperature=temperature,
            stop_sequences=tuple(stop_sequences))).text.strip()
