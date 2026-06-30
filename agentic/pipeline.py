"""Agentic RAG pipeline — the agent loop gathers evidence, then we generate from it.

Two entrypoints:
  * Pipeline.answer(query)   — standalone prod-style usage (builds its own index + graph from common).
  * run(query, bundle, k)    — benchmark adapter (reuses the shared, pre-built index + graph).
"""
from __future__ import annotations

from common import generate
from common.corpus import docs as corpus_docs
from common.index import CHUNKERS, Index
from graphrag import build as build_graph

from .agent import run_agent
from .config import Config


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
        docs, ctx, _ = run_agent(
            query, self._index_(), self._graph_(), self.cfg.max_steps, self.cfg.tool_k)
        return docs, ctx

    def answer(self, query: str) -> str:
        _, ctx = self.retrieve(query)
        return generate.answer(query, ctx)


def run(query, bundle, k: int = 5):
    docs, ctx, steps = run_agent(
        query, bundle.indexes["sentence"], bundle.graph, Config.max_steps, Config.tool_k)
    return docs, ctx, {"steps": steps}
