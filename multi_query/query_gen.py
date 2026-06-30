"""Query generation — the LLM half of multi-query retrieval.

A thin, prod-named seam over common.transform.gen_queries: keeps the call site in the retriever
readable and gives this architecture one place to swap or tune query expansion later.
"""
from __future__ import annotations

from common.transform import gen_queries


def generate_queries(query: str, n: int = 3) -> list[str]:
    """Return the original query plus n LLM-generated rephrasings/broadenings (n+1 total)."""
    return gen_queries(query, n)
