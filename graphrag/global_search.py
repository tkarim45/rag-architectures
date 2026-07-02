"""Global search — corpus-level retrieval via community map-reduce.

Local search cannot answer "what themes run across this corpus" — no seed entity exists to expand
from. The paper's answer is query-focused summarization over community reports:

  * **Map:** every summarized community is independently rated 0-10 for relevance to the question
    (one structured LLM call per community — cheap here, parallelizable at real scale).
  * **Reduce:** keep the top `config.max_communities` communities rated at or above
    `config.min_community_rating`, then rank their provenance documents by the summed rating of
    every kept community that contains them (a doc backed by two relevant communities outranks a
    doc backed by one).

The full paper generates the answer from the community summaries themselves; at 14 documents the
summaries and the documents fit the same context budget, so we keep the shared benchmark contract
(retrieval returns *documents*) and let the ratings decide which documents represent the corpus.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core import LLM, StructuredCaller, StructuredOutputError, Tracer

from .config import Config
from .graph import KnowledgeGraph
from .prompts import COMMUNITY_RATING_PROMPT


def validate_rating(payload: Any) -> int:
    """Validator for the map step: {"score": <0-10 int>} → clamped int."""
    if not isinstance(payload, dict):
        raise TypeError(f"rating must be a JSON object, got {type(payload).__name__}")
    score = payload["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError("score must be a number")
    return max(0, min(10, int(score)))


@dataclass(frozen=True)
class GlobalSearchResult:
    """Ranked provenance docs plus the map-reduce audit trail."""

    doc_ids: tuple[str, ...]                       # ranked, truncated to top_k_docs
    doc_scores: tuple[tuple[str, float], ...]      # (doc_id, summed rating), pre-truncation
    community_ratings: tuple[tuple[int, int], ...]  # (community_id, rating) for every rated one
    communities_consulted: tuple[int, ...]         # ids kept by the reduce step


@dataclass
class GlobalSearch:
    llm: LLM
    graph: KnowledgeGraph
    config: Config
    tracer: Tracer

    def _rate(self, question: str, summary: str) -> int:
        """One map call. A structured-output failure scores the community 0 rather than aborting
        the whole reduce — losing one community is recoverable, losing the query is not."""
        try:
            return StructuredCaller(self.llm).call(
                COMMUNITY_RATING_PROMPT.format(question=question, summary=summary),
                validator=validate_rating, max_tokens=64)
        except StructuredOutputError:
            self.tracer.count("graphrag.rating_failures")
            return 0

    def search(self, question: str) -> GlobalSearchResult:
        with self.tracer.span("graphrag.global_search",
                              communities=len(self.graph.communities)) as span:
            ratings: list[tuple[int, int]] = []
            for community in self.graph.communities:
                if not community.summary:          # below min size — never summarized
                    continue
                with self.tracer.span("graphrag.rate_community",
                                      community_id=community.community_id) as rate_span:
                    rating = self._rate(question, community.summary)
                    rate_span.set("rating", rating)
                ratings.append((community.community_id, rating))

            kept = sorted((r for r in ratings if r[1] >= self.config.min_community_rating),
                          key=lambda item: (-item[1], item[0]))[:self.config.max_communities]
            by_id = {c.community_id: c for c in self.graph.communities}
            scores: dict[str, float] = {}
            for community_id, rating in kept:
                for doc_id in by_id[community_id].doc_ids:
                    scores[doc_id] = scores.get(doc_id, 0.0) + float(rating)
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

            span.set("rated", len(ratings))
            span.set("kept", [community_id for community_id, _ in kept])
            span.set("docs", len(ranked))
        return GlobalSearchResult(
            doc_ids=tuple(doc_id for doc_id, _ in ranked[:self.config.top_k_docs]),
            doc_scores=tuple(ranked),
            community_ratings=tuple(ratings),
            communities_consulted=tuple(community_id for community_id, _ in kept))
