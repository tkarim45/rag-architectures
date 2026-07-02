"""Collapsed-tree retrieval over the RAPTOR tree.

Sarthi et al. evaluate two query strategies — level-by-level tree traversal and the "collapsed
tree" that flattens all levels into one candidate pool — and the collapsed tree performs better
(their Fig. 4 / Sec. 5), so it is the only strategy implemented here. Every node of every level
is scored by cosine similarity against the query embedding; because all embeddings are
L2-normalized at the embedder boundary (see `core.embeddings.base`), one matrix–vector dot
product scores the whole tree. Selection is then greedy by score under a token budget
(`config.max_context_tokens`, approximated as chars/4), matching the paper's "add nodes by
decreasing similarity until the token threshold" procedure.

Node → chunk mapping (the contract seam — read this before changing anything)
------------------------------------------------------------------------------
RAPTOR's unit of retrieval is a *node*, but the core contract speaks in `Chunk`s, and the
benchmark scores architectures by `RetrievalResult.doc_ids` — ranked unique document ids. A
summary node abstracts *several* documents, so mapping each selected node to a single synthetic
chunk (`doc_id = source_doc_ids[0]`) would silently drop the multi-doc credit that is RAPTOR's
whole advantage on multi-hop questions.

The mapping implemented here instead emits, for each selected node in score order, **one
synthetic ScoredChunk per source document**:

    chunk_id = f"{node.node_id}@{doc_id}",  doc_id = doc_id,  display_text = node.text

Consequences, all intentional:
  * `RetrievalResult.doc_ids` (first-occurrence dedup over chunks) naturally yields every
    document a selected summary abstracts, in node-score order — doc-level metrics see the
    full credit.
  * `core.ContextBuilder` deduplicates identical `display_text`s, so the N per-doc chunks of one
    node collapse back into a single context passage — the generator never reads duplicates.
  * `diagnostics["ranked_doc_ids"]` records the same ordered dedup explicitly, so benchmark
    readers don't have to re-derive it from chunk internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core import Chunk, Query, RetrievalResult, ScoredChunk
from core.embeddings import Embedder
from core.telemetry import Tracer, tracer as default_tracer

from .config import Config
from .tree import RaptorNode, RaptorTree

RETRIEVER_NAME = "raptor.collapsed_tree"


@dataclass
class CollapsedTreeRetriever:
    """Scores the whole tree per query and packages selected nodes into contract types."""

    tree: RaptorTree
    embedder: Embedder
    config: Config = field(default_factory=Config)
    tracer: Tracer = field(default_factory=lambda: default_tracer)

    def retrieve(self, question: str) -> RetrievalResult:
        query = Query(text=question, top_k=self.config.top_nodes)
        with self.tracer.span("raptor.retrieve", nodes_in_tree=len(self.tree)) as span:
            ordered_nodes, matrix = self.tree.embedding_matrix()
            if not ordered_nodes:
                span.set("selected", 0)
                return RetrievalResult(query=query, chunks=[],
                                       diagnostics={"selected_nodes": [], "ranked_doc_ids": []})

            with self.tracer.span("raptor.embed_query"):
                query_vec = self.embedder.embed_query(question)
            scores = matrix @ query_vec.astype(np.float32)     # cosine == dot (L2-normalized)

            selected = self._select_under_budget(ordered_nodes, scores)
            chunks = self._to_scored_chunks(selected)
            diagnostics = self._diagnostics(selected, scored_count=len(ordered_nodes))

            span.set("selected", len(selected))
            span.set("levels_in_selection", diagnostics["levels_in_selection"])
            span.set("budget_tokens_used", diagnostics["budget_tokens_used"])
        return RetrievalResult(query=query, chunks=chunks, diagnostics=diagnostics)

    # ---- selection ----------------------------------------------------------------------

    def _select_under_budget(self, nodes: list[RaptorNode],
                             scores: np.ndarray) -> list[tuple[RaptorNode, float]]:
        """Greedy by score under the token budget, capped at `top_nodes`.

        The top-ranked node is always admitted even if it alone exceeds the budget — returning
        an empty context because the best match is a long summary would be strictly worse than
        letting the context builder truncate it.
        """
        ranked = np.argsort(-scores)
        selected: list[tuple[RaptorNode, float]] = []
        used_tokens = 0
        for idx in ranked:
            if len(selected) >= self.config.top_nodes:
                break
            node = nodes[int(idx)]
            cost = node.approx_tokens
            if selected and used_tokens + cost > self.config.max_context_tokens:
                break                                          # budget exhausted (paper's stop)
            selected.append((node, float(scores[int(idx)])))
            used_tokens += cost
        return selected

    # ---- contract mapping ---------------------------------------------------------------

    def _to_scored_chunks(self, selected: list[tuple[RaptorNode, float]]) -> list[ScoredChunk]:
        """One synthetic chunk per (node, source doc) pair — see the module docstring for why."""
        chunks: list[ScoredChunk] = []
        for node, score in selected:
            for doc_id in node.source_doc_ids:
                chunks.append(ScoredChunk(
                    chunk=Chunk(
                        chunk_id=f"{node.node_id}@{doc_id}",
                        doc_id=doc_id,
                        index_text=node.text,
                        display_text=node.text,
                        metadata={"raptor_level": node.level, "raptor_node_id": node.node_id},
                    ),
                    score=score,
                    retriever=RETRIEVER_NAME,
                ))
        return chunks

    def _diagnostics(self, selected: list[tuple[RaptorNode, float]],
                     scored_count: int) -> dict[str, Any]:
        """The retrieval story: which nodes won, at which levels, at what cost.

        `selected_nodes` is the diagnostic that shows RAPTOR working — when a level>=1 summary
        outranks every leaf, that is the multi-hop win (cross-document context in one node) made
        visible.
        """
        ranked_doc_ids: list[str] = []
        seen: set[str] = set()
        for node, _ in selected:
            for doc_id in node.source_doc_ids:
                if doc_id not in seen:
                    seen.add(doc_id)
                    ranked_doc_ids.append(doc_id)
        return {
            "selected_nodes": [
                {"node_id": node.node_id, "level": node.level, "score": round(score, 4),
                 "source_docs": len(node.source_doc_ids), "tokens": node.approx_tokens}
                for node, score in selected
            ],
            "ranked_doc_ids": ranked_doc_ids,
            "levels_in_selection": sorted({node.level for node, _ in selected}),
            "summary_nodes_selected": sum(1 for node, _ in selected if node.level > 0),
            "nodes_scored": scored_count,
            "budget_tokens_used": sum(node.approx_tokens for node, _ in selected),
            "budget_tokens_max": self.config.max_context_tokens,
        }
