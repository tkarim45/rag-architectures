"""Shared retrieval helpers used across architecture packages: Reciprocal Rank Fusion, and the
chunk→doc / chunk→context collapses."""
from __future__ import annotations

from .index import Index


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion over several chunk-id rankings."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return [cid for cid, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def to_docs(cids: list[str], index: Index) -> list[str]:
    """Collapse a chunk ranking to a ranked list of unique doc ids (first occurrence wins)."""
    out: list[str] = []
    for c in cids:
        d = index.doc_of(c)
        if d not in out:
            out.append(d)
    return out


def context(cids: list[str], index: Index, n: int = 5) -> str:
    """Gather the return-texts of the top chunks (dedup) into a context block for generation."""
    seen, blocks = set(), []
    for c in cids:
        t = index.text_of(c)
        if t not in seen:
            seen.add(t)
            blocks.append(t)
        if len(blocks) >= n:
            break
    return "\n\n".join(f"[{i+1}] {b}" for i, b in enumerate(blocks))
