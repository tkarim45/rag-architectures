"""Corrective retriever — the heart of CRAG. Retrieve with hybrid, LLM-grade the top hits, and if
too few clear the relevance bar, rewrite the query and re-retrieve with RAG-Fusion. Returns the
relevant chunk ids plus a flag recording whether the corrective path fired."""
from __future__ import annotations

from hybrid import retrieve as hybrid_retrieve
from rag_fusion import retrieve as fusion_retrieve

from . import grader
from .corrector import rewrite


def retrieve(query: str, index, min_relevant: int = 2) -> tuple[list[str], bool]:
    cids = hybrid_retrieve(query, index, 8)
    relevant = [c for c in cids[:6] if grader.is_relevant(query, index.by_cid[c].index_text)]

    corrected = False
    if len(relevant) < min_relevant:                    # bad first retrieval — self-correct
        corrected = True
        rq = rewrite(query)
        relevant = fusion_retrieve(rq, index, 6)
    return relevant, corrected
