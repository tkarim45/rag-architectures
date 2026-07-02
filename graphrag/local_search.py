"""Local search — entity-centric retrieval, the paper's answer to "who/what/how is X related".

Three stages:

  1. **Entity linking.** The question's entities are extracted by the LLM (structured call) and
     resolved against graph nodes, then *merged* with a lexical fallback that scans for graph
     entity names appearing verbatim in the question. Both run on every query: the LLM catches
     paraphrases the lexical scan misses ("the Estonian founder"), the lexical scan catches
     entities the LLM under-extracts — and it keeps local search alive even when the LLM linker
     fails entirely, which is exactly the degradation the offline test exercises.
  2. **Neighborhood traversal.** Breadth-first expansion from the seed entities up to
     `config.max_hops`, direction-agnostic (a "founded" edge must be walkable from either end).
     Every relation edge whose endpoints were both reached becomes a human-readable path string
     for diagnostics — the trace of *why* a document was retrieved.
  3. **Provenance scoring.** Documents earn score from two sources, both decayed by hop distance:
     each visited entity votes for the docs that mention it with weight 1/(1+hop), and each
     traversed relation votes for the doc that stated it with weight 1/(1+deepest endpoint hop).
     Seeds therefore dominate, bridge docs surface through their edges, and the tail truncates at
     `config.top_k_docs`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core import LLM, StructuredCaller, StructuredOutputError, Tracer

from .config import Config
from .extractor import normalize_entity_name
from .graph import KnowledgeGraph
from .prompts import QUERY_ENTITY_PROMPT


def validate_entity_list(payload: Any) -> tuple[str, ...]:
    """Validator for the query-entity call: JSON array of strings → normalized, deduped names."""
    if not isinstance(payload, list):
        raise TypeError(f"query entities must be a JSON array, got {type(payload).__name__}")
    names: list[str] = []
    for item in payload:
        if not isinstance(item, str):
            raise TypeError("each query entity must be a string")
        name = normalize_entity_name(item)
        if name and name not in names:
            names.append(name)
    return tuple(names)


@dataclass(frozen=True)
class LocalSearchResult:
    """Ranked provenance docs plus the full story of how local search got there."""

    doc_ids: tuple[str, ...]                 # ranked, truncated to top_k_docs
    doc_scores: tuple[tuple[str, float], ...]  # (doc_id, score), ranked, pre-truncation
    matched_entities: tuple[str, ...]        # seed nodes (normalized), llm + lexical merged
    llm_entities: tuple[str, ...]            # what the LLM linker resolved to graph nodes
    lexical_entities: tuple[str, ...]        # what the substring fallback matched
    paths: tuple[str, ...]                   # 'A -[rel]-> B' for every traversed relation
    entity_hops: tuple[tuple[str, int], ...]  # visited entity -> hop distance from a seed


@dataclass
class LocalSearch:
    llm: LLM
    graph: KnowledgeGraph
    config: Config
    tracer: Tracer

    # ---- stage 1: entity linking ---------------------------------------------------------

    def _llm_entities(self, question: str) -> tuple[str, ...]:
        """LLM entity extraction, resolved to graph nodes. Failure is survivable (the lexical
        fallback still runs), so a structured-output failure degrades instead of raising."""
        try:
            candidates = StructuredCaller(self.llm).call(
                QUERY_ENTITY_PROMPT.format(question=question),
                validator=validate_entity_list, max_tokens=256)
        except StructuredOutputError:
            self.tracer.count("graphrag.query_entity_failures")
            return ()
        resolved: list[str] = []
        for candidate in candidates:
            for name in self._resolve(candidate):
                if name not in resolved:
                    resolved.append(name)
        return tuple(resolved)

    def _resolve(self, candidate: str) -> list[str]:
        """Map an extracted entity string to graph nodes: exact match first, then containment
        either way ("quorrel 3.0" should still hit the "quorrel" node and vice versa)."""
        if self.graph.has_entity(candidate):
            return [candidate]
        floor = self.config.entity_match_min_chars
        return [name for name in self.graph.entity_names
                if len(name) >= floor and (name in candidate or candidate in name)]

    def _lexical_entities(self, question: str) -> tuple[str, ...]:
        """Fallback linker: graph entity names appearing verbatim in the (normalized) question."""
        normalized_question = normalize_entity_name(question)
        floor = self.config.entity_match_min_chars
        return tuple(sorted(name for name in self.graph.entity_names
                            if len(name) >= floor and name in normalized_question))

    # ---- stage 2 + 3: traversal and scoring -----------------------------------------------

    def search(self, question: str) -> LocalSearchResult:
        with self.tracer.span("graphrag.local_search", max_hops=self.config.max_hops) as span:
            llm_entities = self._llm_entities(question)
            lexical_entities = self._lexical_entities(question)
            seeds = list(llm_entities)
            for name in lexical_entities:
                if name not in seeds:
                    seeds.append(name)
            span.set("llm_entities", list(llm_entities))
            span.set("lexical_entities", list(lexical_entities))

            hop_of: dict[str, int] = {seed: 0 for seed in seeds}
            frontier = list(seeds)
            for hop in range(1, self.config.max_hops + 1):
                next_frontier: list[str] = []
                for entity in frontier:
                    for neighbor in sorted(self.graph.neighbors(entity)):
                        if neighbor not in hop_of:
                            hop_of[neighbor] = hop
                            next_frontier.append(neighbor)
                frontier = next_frontier

            scores: dict[str, float] = {}
            for entity, hop in hop_of.items():
                for doc_id in self.graph.docs_of(entity):
                    scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (1.0 + hop)

            paths: list[tuple[int, str]] = []
            for source, target, data in self.graph.graph.edges(data=True):
                if source in hop_of and target in hop_of:
                    depth = max(hop_of[source], hop_of[target])
                    doc_id = str(data.get("doc_id", ""))
                    if doc_id:
                        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (1.0 + depth)
                    paths.append((depth, f"{self.graph.display_name(source)} "
                                         f"-[{data.get('type', 'related_to')}]-> "
                                         f"{self.graph.display_name(target)}"))
            paths.sort()

            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            span.set("seeds", seeds)
            span.set("visited", len(hop_of))
            span.set("docs", len(ranked))
        return LocalSearchResult(
            doc_ids=tuple(doc_id for doc_id, _ in ranked[:self.config.top_k_docs]),
            doc_scores=tuple((doc_id, round(score, 4)) for doc_id, score in ranked),
            matched_entities=tuple(seeds),
            llm_entities=llm_entities,
            lexical_entities=lexical_entities,
            paths=tuple(dict.fromkeys(path for _, path in paths)),
            entity_hops=tuple(sorted(hop_of.items(), key=lambda item: (item[1], item[0]))))
