"""Retrieval metrics. All operate on ranked doc-id lists against a gold set — computed at the
document level (not chunk level) because gold labels are documents; any chunking strategy collapses
to doc ids through `RetrievalResult.doc_ids` first."""
from __future__ import annotations

import math
from typing import Sequence


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int = 5) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & set(gold)) / len(gold)


def hit_at_k(ranked: Sequence[str], gold: Sequence[str], k: int = 5) -> bool:
    return bool(set(ranked[:k]) & set(gold))


def precision_at_k(ranked: Sequence[str], gold: Sequence[str], k: int = 5) -> float:
    if k <= 0:
        return 0.0
    return len(set(ranked[:k]) & set(gold)) / k


def mrr(ranked: Sequence[str], gold: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant doc."""
    gold_set = set(gold)
    for i, doc_id in enumerate(ranked):
        if doc_id in gold_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Sequence[str], k: int = 5) -> float:
    """Binary-relevance NDCG@k: rewards putting *all* gold docs high, not just one."""
    gold_set = set(gold)
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in gold_set)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_set), k)))
    return dcg / ideal if ideal > 0 else 0.0
