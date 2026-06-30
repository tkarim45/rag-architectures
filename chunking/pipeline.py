"""Chunking-strategy RAG pipeline — same dense retrieve → generate, over a configurable chunk index.

`sentence_window` returns neighbours, `parent_child` returns the whole parent doc, `contextual`
prepends an LLM context blurb before embedding. The retriever never changes; the index granularity
does.

Three entrypoints:
  * Pipeline.answer(query)   — standalone prod-style usage (builds its own index from common).
  * run(query, bundle, k)    — benchmark adapter over the default chunker (Config.chunker).
  * run_for(chunker)         — factory: a run-closure bound to a specific chunker's index, tagging
                               meta with which chunker it used (the benchmark calls this per variant).
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
        cids = retrieve(query, idx, self.cfg.top_k)
        return to_docs(cids, idx), context(cids, idx, self.cfg.return_n)

    def answer(self, query: str) -> str:
        _, ctx = self.retrieve(query)
        return generate.answer(query, ctx)


def run(query, bundle, k: int = 5):
    idx = bundle.indexes[Config.chunker]
    cids = retrieve(query, idx, max(k, 8))
    return to_docs(cids, idx), context(cids, idx, Config.return_n), {"chunker": Config.chunker}


def run_for(chunker: str):
    """Build a `run(query, bundle, k)` closure bound to one chunker's pre-built index in the bundle."""
    def run(query, bundle, k: int = 5):
        idx = bundle.indexes[chunker]
        cids = retrieve(query, idx, max(k, 8))
        return to_docs(cids, idx), context(cids, idx, Config.return_n), {"chunker": chunker}
    return run
