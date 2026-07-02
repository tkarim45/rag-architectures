"""CorpusIndex — the query-time handle every architecture retrieves against.

Built once, offline, by `IngestionPipeline`: documents → chunks → embeddings → vector store, plus
a BM25 index over the same chunks. Online code gets typed search methods that return `ScoredChunk`
(never raw ids), so provenance flows through every architecture for free.

The offline/online split is deliberate and mirrors production: index builders write here at deploy
time; retrievers only read.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..embeddings.base import Embedder
from ..errors import IndexError_
from ..stores.lexical import BM25Index
from ..stores.vector import VectorStore, build_vector_store
from ..telemetry import Tracer, tracer as default_tracer
from ..types import Chunk, Document, ScoredChunk
from .chunkers import Chunker


class CorpusIndex:
    def __init__(self, *, documents: Sequence[Document], chunks: Sequence[Chunk],
                 embedder: Embedder, vector_store: VectorStore, bm25: BM25Index,
                 strategy: str = ""):
        self.strategy = strategy
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25
        self._documents = {d.doc_id: d for d in documents}
        self._chunks = {c.chunk_id: c for c in chunks}
        if len(self._chunks) != len(chunks):
            raise IndexError_("duplicate chunk ids in corpus")

    # ---- lookups -----------------------------------------------------------------------

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    @property
    def documents(self) -> list[Document]:
        return list(self._documents.values())

    def chunk(self, chunk_id: str) -> Chunk:
        return self._chunks[chunk_id]

    def document(self, doc_id: str) -> Document:
        return self._documents[doc_id]

    def chunks_of(self, doc_id: str) -> list[Chunk]:
        return [c for c in self._chunks.values() if c.doc_id == doc_id]

    # ---- search ------------------------------------------------------------------------

    def dense_search(self, query: str, k: int) -> list[ScoredChunk]:
        return self.dense_search_vector(self.embedder.embed_query(query), k)

    def dense_search_vector(self, query_vector: np.ndarray, k: int) -> list[ScoredChunk]:
        hits = self.vector_store.search(query_vector, k)
        return [ScoredChunk(self._chunks[h.id], h.score, retriever="dense") for h in hits]

    def sparse_search(self, query: str, k: int) -> list[ScoredChunk]:
        hits = self.bm25.search(query, k)
        return [ScoredChunk(self._chunks[h.id], h.score, retriever="sparse") for h in hits]


@dataclass
class IngestionPipeline:
    """Offline build: chunk → embed → write both index halves. One pipeline per chunking strategy;
    the benchmark builds several and shares them across architectures so comparisons stay honest."""

    chunker: Chunker
    embedder: Embedder
    vector_backend: str = "faiss"
    tracer: Tracer | None = None

    def ingest(self, documents: Sequence[Document]) -> CorpusIndex:
        trace = self.tracer or default_tracer
        with trace.span("ingest", strategy=self.chunker.name, docs=len(documents)) as span:
            chunks = self.chunker.chunk(documents)
            if not chunks:
                raise IndexError_("chunker produced no chunks")
            vectors = self.embedder.embed_texts([c.index_text for c in chunks])
            store = build_vector_store(self.vector_backend)
            store.add([c.chunk_id for c in chunks], vectors,
                      [dict(c.metadata, doc_id=c.doc_id) for c in chunks])
            bm25 = BM25Index()
            bm25.add([c.chunk_id for c in chunks], [c.index_text for c in chunks])
            span.set("chunks", len(chunks))
        return CorpusIndex(documents=documents, chunks=chunks, embedder=self.embedder,
                           vector_store=store, bm25=bm25, strategy=self.chunker.name)
