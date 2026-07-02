"""Online orchestration: route the question, run the right search, resolve docs to chunks.

The retriever is the seam between GraphRAG's graph world (entity names, hops, community ratings)
and the framework's retrieval contract (`RetrievalResult` of `ScoredChunk`s). Ranked provenance
doc ids come out of local/global search; the whole-document index turns each into its single
whole-doc chunk so the generator reads full documents — the graph decided *which* documents,
provenance decided *why*, and both stories land in `diagnostics`.

Routing: `config.mode` pins the search when the caller knows the workload; `"auto"` spends one
tiny structured call asking whether the question names specific entities (→ local) or a broad
corpus theme (→ global), defaulting to local on any failure because entity questions dominate
this dataset and local search degrades more gracefully.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core import (CorpusIndex, LLM, Query, RetrievalResult, ScoredChunk, StructuredCaller,
                  StructuredOutputError, Tracer)

from .config import Config
from .global_search import GlobalSearch, GlobalSearchResult
from .graph import KnowledgeGraph
from .local_search import LocalSearch, LocalSearchResult
from .prompts import ROUTER_PROMPT


def validate_mode(payload: Any) -> str:
    """Validator for the router call: {"mode": "local"|"global"} → the mode string."""
    if not isinstance(payload, dict):
        raise TypeError(f"router reply must be a JSON object, got {type(payload).__name__}")
    mode = str(payload["mode"]).strip().casefold()
    if mode not in ("local", "global"):
        raise ValueError(f"router mode must be 'local' or 'global', got {mode!r}")
    return mode


@dataclass
class GraphRetriever:
    llm: LLM
    graph: KnowledgeGraph
    index: CorpusIndex
    config: Config
    tracer: Tracer
    _local: LocalSearch = field(init=False)
    _global: GlobalSearch = field(init=False)

    def __post_init__(self) -> None:
        self._local = LocalSearch(llm=self.llm, graph=self.graph, config=self.config,
                                  tracer=self.tracer)
        self._global = GlobalSearch(llm=self.llm, graph=self.graph, config=self.config,
                                    tracer=self.tracer)

    # ---- routing -----------------------------------------------------------------------

    def _route(self, question: str) -> tuple[str, str]:
        """Pick the search mode; returns (mode, how_it_was_decided) for diagnostics."""
        if self.config.mode != "auto":
            return self.config.mode, "config"
        try:
            mode = StructuredCaller(self.llm).call(
                ROUTER_PROMPT.format(question=question), validator=validate_mode, max_tokens=32)
            return mode, "llm"
        except StructuredOutputError:
            self.tracer.count("graphrag.router_failures")
            return "local", "fallback"

    # ---- retrieval ---------------------------------------------------------------------

    def retrieve(self, question: str) -> RetrievalResult:
        with self.tracer.span("graphrag.retrieve", mode=self.config.mode) as span:
            mode, decided_by = self._route(question)
            span.set("selected_mode", mode)
            span.set("routed_by", decided_by)

            diagnostics: dict[str, Any] = {
                "architecture": "graphrag",
                "mode": mode,
                "router": {"configured": self.config.mode, "selected": mode,
                           "decided_by": decided_by},
                "graph": self.graph.stats(),
            }
            if mode == "local":
                local = self._local.search(question)
                doc_ids, doc_scores = local.doc_ids, dict(local.doc_scores)
                diagnostics.update(self._local_diagnostics(local))
            else:
                global_ = self._global.search(question)
                doc_ids, doc_scores = global_.doc_ids, dict(global_.doc_scores)
                diagnostics.update(self._global_diagnostics(global_))

            chunks = self._to_chunks(doc_ids, doc_scores, retriever=f"graphrag.{mode}")
            span.set("docs", len(doc_ids))
            span.set("chunks", len(chunks))
        return RetrievalResult(query=Query(text=question, top_k=self.config.top_k_docs),
                               chunks=chunks, diagnostics=diagnostics)

    def _to_chunks(self, doc_ids: tuple[str, ...], doc_scores: dict[str, float], *,
                   retriever: str) -> list[ScoredChunk]:
        """Ranked doc ids → their whole-document chunks, preserving graph rank order and carrying
        the graph score so downstream fusion/inspection sees real numbers, not rank guesses."""
        chunks: list[ScoredChunk] = []
        for doc_id in doc_ids:
            for chunk in self.index.chunks_of(doc_id):
                chunks.append(ScoredChunk(chunk=chunk, score=doc_scores.get(doc_id, 0.0),
                                          retriever=retriever))
        return chunks

    # ---- diagnostics --------------------------------------------------------------------

    @staticmethod
    def _local_diagnostics(result: LocalSearchResult) -> dict[str, Any]:
        return {
            "matched_entities": list(result.matched_entities),
            "entity_linking": {"llm": list(result.llm_entities),
                               "lexical": list(result.lexical_entities)},
            "traversal_paths": list(result.paths),
            "entity_hops": dict(result.entity_hops),
            "doc_scores": dict(result.doc_scores),
            "seeded": bool(result.matched_entities),
        }

    @staticmethod
    def _global_diagnostics(result: GlobalSearchResult) -> dict[str, Any]:
        return {
            "community_ratings": dict(result.community_ratings),
            "communities_consulted": list(result.communities_consulted),
            "doc_scores": dict(result.doc_scores),
        }
