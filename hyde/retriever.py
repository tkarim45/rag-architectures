"""HyDE retriever — draft a hypothetical answer, embed *that* (not the question), then dense
retrieve. Answer-shaped text lands near real answer passages, bridging the query→document gap."""
from __future__ import annotations

from common import providers

from .hypothesis import generate_hypothesis


def retrieve(query: str, index, k: int = 8) -> list[str]:
    doc = generate_hypothesis(query)               # LLM drafts a hypothetical answer
    dvec = providers.embed([doc])[0]               # embed the answer, not the question
    return [cid for cid, _ in index.dense(dvec, k)]
