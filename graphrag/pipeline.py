"""GraphRAG pipeline — orchestrates build → retrieve → generate.

The two halves of GraphRAG are clearly split:
  * OFFLINE (build, once): extract entities per doc and assemble the shared-entity doc-doc graph.
  * ONLINE  (per query):   seed from query entities, BFS-traverse the graph, generate from context.

Two entrypoints:
  * Pipeline.answer(query)   — standalone prod-style usage (builds its own graph from common).
  * run(query, bundle, k)    — benchmark adapter (reuses the shared, pre-built graph in the bundle).
"""
from __future__ import annotations

from common import generate
from common.corpus import docs as corpus_docs

from .config import Config
from .graph_builder import build
from .retriever import retrieve


class Pipeline:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self._graph = None

    def _graph_(self):
        if self._graph is None:                       # offline build, lazily, once
            self._graph = build(corpus_docs())
        return self._graph

    def retrieve(self, query: str):
        return retrieve(query, self._graph_(), self.cfg.top_k, self.cfg.hops)

    def answer(self, query: str) -> str:
        _, ctx = self.retrieve(query)
        return generate.answer(query, ctx)


def run(query, bundle, k: int = 5):
    docs, ctx = retrieve(query, bundle.graph, k)
    return docs, ctx, {}
