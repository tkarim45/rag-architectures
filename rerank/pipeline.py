"""Rerank RAG pipeline — orchestrates dense recall → cross-encoder rerank → generate.

Two-stage retrieval: a bi-encoder embeds the query and recalls a generous candidate set (fast,
coarse), then a cross-encoder scores each `(query, chunk)` pair jointly and reorders for precision.
The reordered top-k becomes the context handed to the generator.

Two entrypoints:
  * Pipeline.answer(query)   — standalone prod-style usage (builds its own index from common).
  * run(query, bundle, k)    — benchmark adapter (reuses the shared, pre-built index in the bundle).
"""
from __future__ import annotations

from common import generate
from common.corpus import docs as corpus_docs
from common.index import CHUNKERS, Index
from common.retrieval import context, to_docs

from .config import Config
from .retriever import retrieve


class Pipeline:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self._index = None

    def _index_(self):
        if self._index is None:                         # offline indexing, lazily, once
            self._index = Index(CHUNKERS[self.cfg.chunker](corpus_docs()))
        return self._index

    def retrieve(self, query: str):
        idx = self._index_()
        cids = retrieve(query, idx, self.cfg.top_k, self.cfg.recall_k, self.cfg.model)
        return to_docs(cids, idx), context(cids, idx, self.cfg.return_n)

    def answer(self, query: str) -> str:
        _, ctx = self.retrieve(query)
        return generate.answer(query, ctx)


def run(query, bundle, k: int = 5):
    idx = bundle.indexes[Config.chunker]
    cids = retrieve(query, idx, max(k, Config.top_k), Config.recall_k, Config.model)
    return to_docs(cids, idx), context(cids, idx, Config.return_n), {}
