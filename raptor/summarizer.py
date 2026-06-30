"""Cluster summarization — the offline half of RAPTOR's tree build. Each cluster of related leaf
documents is condensed by the LLM into a short abstractive summary that becomes a parent node, so a
single retrieval hit on the summary can pull in every document it covers."""
from __future__ import annotations

from common import providers


def summarize(texts: list[str]) -> str:
    return providers.complete(
        "Summarize the shared topic and key facts across these documents in 2-3 sentences:\n"
        + "\n\n".join(texts),
        max_tokens=200,
    )
