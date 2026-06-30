"""Multi-query retriever — expand the query into variants, dense-retrieve each, union by best rank.

Each query variant produces its own dense top-k ranking. We union those rankings, keeping each
chunk's BEST (lowest) rank across all variants, then return the top-k chunks by that best rank. A
chunk that ranks #1 for any single phrasing wins, so broadening the query can only help recall.
"""
from __future__ import annotations

from common import providers

from .query_gen import generate_queries


def retrieve(query: str, index, k: int = 8, n_queries: int = 3) -> list[str]:
    best_rank: dict[str, int] = {}
    for q in generate_queries(query, n_queries):
        qvec = providers.embed([q])[0]
        for rank, (cid, _) in enumerate(index.dense(qvec, k)):
            if cid not in best_rank or rank < best_rank[cid]:
                best_rank[cid] = rank
    ranked = sorted(best_rank, key=lambda cid: best_rank[cid])
    return ranked[:k]
