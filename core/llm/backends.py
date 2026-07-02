"""Concrete LLM backends: Claude on AWS Bedrock (default) and the direct Anthropic API.

Both normalize SDK exceptions into the framework's error taxonomy so retry policy lives in one
place (`BaseLLM`) and pipelines never import botocore/anthropic error types.
"""
from __future__ import annotations

from ..config import LLMConfig
from ..errors import ConfigurationError, ProviderError, RateLimitError
from ..types import TokenUsage
from .base import BaseLLM, Completion, CompletionRequest


def _classify(provider: str, exc: Exception) -> ProviderError:
    """Map an SDK exception onto our taxonomy. Anything that smells like throttling or a transient
    service fault is retryable; auth/validation errors are not."""
    name = type(exc).__name__
    text = str(exc)
    if "Throttling" in name or "RateLimit" in name or "429" in text:
        return RateLimitError(f"{name}: {exc}", provider=provider)
    transient = ("Timeout" in name or "ServiceUnavailable" in name or "InternalServer" in name
                 or "overloaded" in text.lower() or "503" in text or "500" in text)
    return ProviderError(f"{name}: {exc}", provider=provider, retryable=transient)


class BedrockClaudeLLM(BaseLLM):
    """Claude via `anthropic[bedrock]`. Credentials come from the standard AWS chain
    (env vars / profile / role) — loaded from `.env` + `~/.env` by core.config at import."""

    def __init__(self, config: LLMConfig):
        super().__init__(retry=config.retry)
        self.config = config
        self._client = None

    def _client_(self):
        if self._client is None:
            try:
                from anthropic import AnthropicBedrock
            except ImportError as e:  # pragma: no cover
                raise ConfigurationError(
                    "anthropic[bedrock] is not installed — `pip install 'anthropic[bedrock]'`") from e
            self._client = AnthropicBedrock(
                aws_region=self.config.aws_region, timeout=self.config.request_timeout_s)
        return self._client

    def _invoke(self, request: CompletionRequest) -> Completion:
        kwargs: dict = dict(
            model=self.config.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            messages=[{"role": "user", "content": request.prompt}],
        )
        if request.system:
            kwargs["system"] = request.system
        if request.stop_sequences:
            kwargs["stop_sequences"] = list(request.stop_sequences)
        try:
            msg = self._client_().messages.create(**kwargs)
        except Exception as e:  # noqa: BLE001 — SDK raises many types; classified below
            raise _classify("bedrock", e) from e
        text = "".join(b.text for b in msg.content if b.type == "text")
        return Completion(
            text=text,
            usage=TokenUsage(msg.usage.input_tokens, msg.usage.output_tokens),
            model=self.config.model,
            stop_reason=msg.stop_reason or "")


class AnthropicLLM(BaseLLM):
    """Claude via the direct Anthropic API (`ANTHROPIC_API_KEY`). Same normalization as Bedrock."""

    def __init__(self, config: LLMConfig):
        super().__init__(retry=config.retry)
        self.config = config
        self._client = None

    def _client_(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as e:  # pragma: no cover
                raise ConfigurationError("anthropic is not installed") from e
            self._client = Anthropic(timeout=self.config.request_timeout_s)
        return self._client

    def _invoke(self, request: CompletionRequest) -> Completion:
        kwargs: dict = dict(
            model=self.config.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            messages=[{"role": "user", "content": request.prompt}],
        )
        if request.system:
            kwargs["system"] = request.system
        if request.stop_sequences:
            kwargs["stop_sequences"] = list(request.stop_sequences)
        try:
            msg = self._client_().messages.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            raise _classify("anthropic", e) from e
        text = "".join(b.text for b in msg.content if b.type == "text")
        return Completion(
            text=text,
            usage=TokenUsage(msg.usage.input_tokens, msg.usage.output_tokens),
            model=self.config.model,
            stop_reason=msg.stop_reason or "")


def build_llm(config: LLMConfig) -> BaseLLM:
    if config.backend == "bedrock":
        return BedrockClaudeLLM(config)
    if config.backend == "anthropic":
        return AnthropicLLM(config)
    raise ConfigurationError(f"unknown LLM backend {config.backend!r}")
