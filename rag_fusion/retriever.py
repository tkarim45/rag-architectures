"""RAG-Fusion retriever — generate several query reformulations, dense-retrieve a ranking for each,
then fuse ALL rankings with Reciprocal Rank Fusion. The RRF consensus across reformulations is what
distinguishes this from multi_query's flat union: chunks that surface for *multiple* phrasings rise
to the top, and a single off rephrasing dragging in junk gets outvoted."""
from __future__ import annotations

from common import providers
from common.retrieval import rrf

from .query_gen import generate_queries


def retrieve(query: str, index, k: int = 8, n_queries: int = 3) -> list[str]:
    queries = generate_queries(query, n_queries)
    vecs = providers.embed(queries)
    rankings = [[cid for cid, _ in index.dense(vec, k)] for vec in vecs]
    return rrf(rankings)[:k]
