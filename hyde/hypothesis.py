"""Hypothesis generation — the offline-document imitation step of HyDE.

Makes `n_hypotheses` independent LLM calls (not one call asking for n passages) so each hypothesis
is a clean sample: independent calls at temperature > 0 give genuinely diverse drafts, and a
malformed response poisons at most one hypothesis instead of the whole batch.
"""
from __future__ import annotations

from core import CompletionRequest, LLM, Tracer

from .config import Config
from .prompts import HYPOTHESIS_SYSTEM, hypothesis_prompt


def generate_hypotheses(llm: LLM, question: str, config: Config, tracer: Tracer) -> list[str]:
    """Generate `config.n_hypotheses` hypothetical documents for `question`.

    Returns the non-empty hypotheses in generation order. May return fewer than requested (empty
    completions are dropped rather than embedded as zero-signal noise); the retriever falls back
    to plain query search when the list comes back empty, so a misbehaving LLM degrades HyDE to
    naive dense retrieval instead of crashing the pipeline.
    """
    hypotheses: list[str] = []
    with tracer.span("hyde.hypotheses", n=config.n_hypotheses,
                     temperature=config.temperature) as span:
        for _ in range(config.n_hypotheses):
            completion = llm.complete(CompletionRequest(
                prompt=hypothesis_prompt(question),
                system=HYPOTHESIS_SYSTEM,
                max_tokens=config.hypothesis_max_tokens,
                temperature=config.temperature,
            ))
            text = completion.text.strip()
            if text:
                hypotheses.append(text)
        span.set("generated", len(hypotheses))
    return hypotheses
