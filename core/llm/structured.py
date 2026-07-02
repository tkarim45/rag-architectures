"""Structured output over a plain completion API: ask for JSON, validate, repair, retry.

LLM stages that feed *code* (graders, routers, extractors, planners) must not depend on the model
formatting a blob correctly on the first try. `StructuredCaller` extracts the first JSON value from
the response (models love to wrap JSON in prose or fences), validates it against a caller-supplied
checker, and on failure re-prompts once with the parse error attached — the cheapest reliable
repair loop. After `max_attempts` it raises `StructuredOutputError` with the raw text preserved.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, TypeVar

from ..errors import StructuredOutputError
from .base import LLM, CompletionRequest

T = TypeVar("T")

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Pull the first JSON value out of `text`: try fenced block, then the first {...} or [...]
    balanced region, then the raw text."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    raise json.JSONDecodeError("no JSON value found", text, 0)


class StructuredCaller:
    def __init__(self, llm: LLM, *, max_attempts: int = 2):
        self.llm = llm
        self.max_attempts = max_attempts

    def call(self, prompt: str, *, validator: Callable[[Any], T] | None = None,
             system: str | None = None, max_tokens: int = 1024) -> T:
        """Run `prompt`, parse JSON from the reply, pass it through `validator` (which should raise
        ValueError/KeyError/TypeError on bad shape and return the typed value)."""
        attempt_prompt = prompt
        last_raw = ""
        for _ in range(self.max_attempts):
            raw = self.llm.complete(CompletionRequest(
                prompt=attempt_prompt, system=system, max_tokens=max_tokens)).text
            last_raw = raw
            try:
                value = extract_json(raw)
                return validator(value) if validator else value
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                attempt_prompt = (
                    f"{prompt}\n\nYour previous reply could not be used: {e}\n"
                    f"Previous reply:\n{raw}\n\n"
                    "Reply again with ONLY the corrected JSON, no prose, no code fences.")
        raise StructuredOutputError(
            f"no valid structured output after {self.max_attempts} attempts", raw=last_raw)
