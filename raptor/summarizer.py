"""LLM summarization of one cluster into the next level's node text.

The summary *is* the abstraction step of RAPTOR: whatever facts survive it are the only facts a
higher-level match can surface, so the prompt (see `prompts.py`) insists on preserving entities
and relationships verbatim. Summaries are generated at temperature 0 — tree construction must be
reproducible for the benchmark, which builds the tree once and shares it.
"""
from __future__ import annotations

from core import CompletionRequest, LLM
from core.telemetry import Tracer

from .config import Config
from .prompts import SUMMARIZE_SYSTEM, summarize_prompt


def summarize_cluster(llm: LLM, texts: list[str], config: Config, tracer: Tracer) -> str:
    """Compress the member node texts of one cluster into a single summary paragraph."""
    with tracer.span("raptor.summarize", members=len(texts)) as span:
        completion = llm.complete(CompletionRequest(
            prompt=summarize_prompt(texts),
            system=SUMMARIZE_SYSTEM,
            max_tokens=config.summary_max_tokens,
            temperature=0.0,
        ))
        summary = completion.text.strip()
        span.set("summary_chars", len(summary))
    return summary
