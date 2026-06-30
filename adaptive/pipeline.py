"""Adaptive RAG pipeline — classify the query, then dispatch to the cheapest sufficient retriever.

A lightweight LLM router labels each query and routes it: simple → dense, multi_hop → GraphRAG,
broad → RAG-Fusion. The bet is that most queries are easy and only the hard ones need the expensive
graph/fusion paths; the risk is router error (a misrouted multi-hop lands on dense and fails).

Two entrypoints:
  * Pipeline.answer(query)   — standalone prod-style usage (builds its own index + graph from common).
  * run(query, bundle, k)    — benchmark adapter (reuses the shared, pre-built index + graph).
"""
from __future__ import annotations

from common import generate
from common.corpus import docs as corpus_docs
from common.index import CHUNKERS, Index
from common.retrieval import context, to_docs
from graphrag import build as build_graph
from graphrag import retrieve as graph_retrieve
from naive import retrieve as dense_retrieve
from rag_fusion import retrieve as fusion_retrieve

from . import router
from .config import Config


def dispatch(query, index, graph, k: int = 5) -> tuple[list[str], str, str]:
    """Route `query`, run the chosen retriever, and return (docs, context, route_label)."""
    r = router.route(query)
    if "multi" in r:                                    # multi_hop → GraphRAG traversal (doc-level)
        docs, ctx = graph_retrieve(query, graph, k)
    elif "broad" in r:                                  # broad → RAG-Fusion consensus retrieval
        cids = fusion_retrieve(query, index, 6)
        docs, ctx = to_docs(cids, index), context(cids, index)
    else:                                               # simple → plain dense retrieval
        cids = dense_retrieve(query, index, k)
        docs, ctx = to_docs(cids, index), context(cids, index)
    return docs, ctx, r


class Pipeline:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self._index = None
        self._graph = None

    def _index_(self):
        if self._index is None:                         # offline indexing, lazily, once
            self._index = Index(CHUNKERS[self.cfg.chunker](corpus_docs()))
        return self._index

    def _graph_(self):
        if self._graph is None:                         # offline graph build, lazily, once
            self._graph = build_graph(corpus_docs())
        return self._graph

    def retrieve(self, query: str):
        docs, ctx, _ = dispatch(query, self._index_(), self._graph_(), self.cfg.top_k)
        return docs, ctx

    def answer(self, query: str) -> str:
        _, ctx = self.retrieve(query)
        return generate.answer(query, ctx)


def run(query, bundle, k: int = 5):
    docs, ctx, r = dispatch(query, bundle.indexes["sentence"], bundle.graph, k)
    return docs, ctx, {"route": r}
