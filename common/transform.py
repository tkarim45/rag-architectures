"""Shared query-transformation helper (used by multi_query and rag_fusion)."""
from __future__ import annotations

import re

from . import providers


def gen_queries(query: str, n: int = 3) -> list[str]:
    """Original query + n LLM-generated rephrasings/broadenings."""
    out = providers.complete(
        f"Generate {n} alternative search queries that rephrase or broaden this question, one per "
        f"line, no numbering:\n{query}", max_tokens=120)
    qs = [re.sub(r"^[\d\.\-\)\s]+", "", ln).strip() for ln in out.splitlines() if ln.strip()]
    return ([query] + qs)[: n + 1]
