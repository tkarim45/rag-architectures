"""HyDE retrieval — vector mixing and dense search.

The core move: search the dense index with a vector built from *hypothetical documents* instead of
(or blended with) the raw question. Document→document similarity is an easier embedding task than
question→document similarity, so a passage-shaped probe lands nearer the true passage.

Mixing (Gao et al. 2022, §3): the paper's InstructGPT + Contriever setup averages the query vector
with the hypothesis vectors. We expose that as a continuous `query_weight`:

    search_vec = l2_normalize(query_weight * q  +  (1 - query_weight) * mean(hypotheses))

`query_weight = 0` is paper-pure HyDE; `query_weight = 1` is naive dense retrieval. The final
L2-renormalization matters because every embedding in the framework is unit-norm and the vector
store assumes cosine == inner product — a shrunken mixed vector would silently deflate scores.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from core import CorpusIndex, Query, RetrievalResult, Tracer
from core.embeddings import Embedder, l2_normalize

from .config import Config


def build_search_vector(embedder: Embedder, question: str, hypotheses: Sequence[str],
                        query_weight: float) -> np.ndarray:
    """Blend the real query vector with the mean hypothesis vector, unit-normalized.

    With no hypotheses (LLM returned nothing usable) this degrades to the plain query vector, i.e.
    naive dense retrieval — the safe fallback, not an exception.
    """
    query_vec = embedder.embed_query(question)
    if not hypotheses:
        return query_vec  # already L2-normalized by the embedder contract
    hypothesis_mean = np.asarray(embedder.embed_texts(list(hypotheses)),
                                 dtype=np.float32).mean(axis=0)
    mixed = query_weight * query_vec + (1.0 - query_weight) * hypothesis_mean
    return l2_normalize(mixed)


def retrieve(index: CorpusIndex, question: str, hypotheses: Sequence[str], config: Config,
             tracer: Tracer) -> RetrievalResult:
    """Dense-search the index with the mixed HyDE vector and package the full story.

    Diagnostics carry everything a benchmark or trace viewer needs to audit the run: the exact
    hypotheses that steered the search (the first thing to read when retrieval goes off-corpus),
    the mixing weight, and whether the fallback path fired.
    """
    with tracer.span("hyde.retrieve", top_k=config.top_k, hypotheses=len(hypotheses),
                     query_weight=config.query_weight) as span:
        search_vector = build_search_vector(index.embedder, question, hypotheses,
                                            config.query_weight)
        hits = index.dense_search_vector(search_vector, config.top_k)
        span.set("chunks", len(hits))
    return RetrievalResult(
        query=Query(text=question, top_k=config.top_k, variants=tuple(hypotheses)),
        chunks=hits,
        diagnostics={
            "hypotheses": list(hypotheses),
            "n_hypotheses_requested": config.n_hypotheses,
            "query_weight": config.query_weight,
            "used_query_fallback": not hypotheses,
        },
    )
