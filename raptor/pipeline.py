"""RAPTOR pipeline — orchestrates build (offline tree) → retrieve → generate.

Two entrypoints:
  * Pipeline.answer(query)   — standalone prod-style usage (builds its own tree from common).
  * run(query, bundle, k)    — benchmark adapter (reuses the shared, pre-built tree in the bundle).
"""
from __future__ import annotations

from common import generate
from common.corpus import docs as corpus_docs

from .config import Config
from .retriever import retrieve
from .tree import build


class Pipeline:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self._raptor = None

    def _raptor_(self):
        if self._raptor is None:                        # offline tree build, lazily, once
            self._raptor = build(corpus_docs(), self.cfg.n_clusters)
        return self._raptor

    def retrieve(self, query: str):
        return retrieve(query, self._raptor_(), self.cfg.top_k)

    def answer(self, query: str) -> str:
        _, ctx = self.retrieve(query)
        return generate.answer(query, ctx)


def run(query, bundle, k: int = 5):
    docs, ctx = retrieve(query, bundle.raptor, k)
    return docs, ctx, {}
