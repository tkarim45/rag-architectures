"""Scoring. Two complementary metrics:
  * retrieval recall@k / hit@k — did the gold docs make it into the top-k? (measures the retriever)
  * answer correctness — LLM-judged against the reference answer (measures the end-to-end system)
"""
from __future__ import annotations


def recall_at_k(ranked_docs: list[str], gold: list[str], k: int = 5) -> float:
    top = set(ranked_docs[:k])
    return len(top & set(gold)) / len(gold) if gold else 0.0


def hit_at_k(ranked_docs: list[str], gold: list[str], k: int = 5) -> bool:
    return bool(set(ranked_docs[:k]) & set(gold))
