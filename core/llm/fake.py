"""Deterministic fake LLM for offline tests.

Rules are (predicate, responder) pairs matched in registration order against the prompt; the first
match wins. The default response echoes a fixed string. Every request is recorded so tests can
assert on *what* a pipeline asked, not just what it returned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..types import TokenUsage
from .base import Completion, CompletionRequest

Predicate = Callable[[CompletionRequest], bool]
Responder = Callable[[CompletionRequest], str]


@dataclass
class FakeLLM:
    default_response: str = "FAKE"
    rules: list[tuple[Predicate, Responder]] = field(default_factory=list)
    requests: list[CompletionRequest] = field(default_factory=list)
    call_count: int = 0
    total_usage: TokenUsage = TokenUsage()

    def on(self, substring: str, response: str | Responder) -> "FakeLLM":
        """Route prompts containing `substring` to `response` (a string or a responder fn)."""
        responder = response if callable(response) else (lambda _req, _r=response: _r)
        self.rules.append((lambda req, s=substring: s in req.prompt, responder))
        return self

    def complete(self, request: CompletionRequest) -> Completion:
        self.requests.append(request)
        self.call_count += 1
        text = self.default_response
        for predicate, responder in self.rules:
            if predicate(request):
                text = responder(request)
                break
        usage = TokenUsage(len(request.prompt) // 4, len(text) // 4)
        self.total_usage = self.total_usage + usage
        return Completion(text=text, usage=usage, model="fake")

    def complete_text(self, prompt: str, *, system: str | None = None, max_tokens: int = 512,
                      temperature: float = 0.0, stop_sequences=()) -> str:
        return self.complete(CompletionRequest(
            prompt=prompt, system=system, max_tokens=max_tokens, temperature=temperature,
            stop_sequences=tuple(stop_sequences))).text.strip()
