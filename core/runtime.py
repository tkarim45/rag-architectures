"""Composition root.

`Runtime` wires the concrete backends (LLM, embedder, caches, tracer) from `CoreConfig` exactly
once; everything downstream receives dependencies by injection. Architecture packages must never
construct backends themselves — that discipline is what lets the test suite run the entire
framework offline with `Runtime.for_testing()` and lets the benchmark share one embedder/LLM
across all thirteen architectures.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

from .config import CoreConfig
from .dataset import documents as dataset_documents
from .embeddings import CachingEmbedder, Embedder, HashingEmbedder, SentenceTransformerEmbedder
from .ingestion import CorpusIndex, IngestionPipeline, build_chunker
from .llm import CachingLLM, FakeLLM, LLM, build_llm
from .telemetry import Tracer, tracer as default_tracer
from .types import Document


@dataclass
class Runtime:
    config: CoreConfig
    llm: LLM
    embedder: Embedder
    tracer: Tracer

    # ---- factories ---------------------------------------------------------------------

    @classmethod
    def from_env(cls, config: CoreConfig | None = None) -> "Runtime":
        """Production wiring: Claude (Bedrock or direct API) + sentence-transformers, with disk
        caches per CacheConfig."""
        config = config or CoreConfig.from_env()
        llm: LLM = build_llm(config.llm)
        if config.cache.llm:
            llm = CachingLLM(llm, config.cache.dir, model_hint=config.llm.model)
        embedder: Embedder = SentenceTransformerEmbedder(
            config.embedding.model, config.embedding.batch_size)
        if config.cache.embeddings:
            embedder = CachingEmbedder(embedder, config.cache.dir,
                                       model_name=config.embedding.model)
        return cls(config=config, llm=llm, embedder=embedder, tracer=default_tracer)

    @classmethod
    def for_testing(cls, llm: LLM | None = None, embedder: Embedder | None = None) -> "Runtime":
        """Offline wiring: FakeLLM + HashingEmbedder + NumPy store. No network, no model download."""
        import os

        os.environ.setdefault("RAGARCH_VECTORSTORE", "numpy")
        config = CoreConfig.from_env()
        return cls(config=config, llm=llm or FakeLLM(), embedder=embedder or HashingEmbedder(),
                   tracer=Tracer())

    # ---- corpus helpers ----------------------------------------------------------------

    @cached_property
    def corpus(self) -> list[Document]:
        return dataset_documents()

    def build_index(self, chunker_name: str = "sentence",
                    documents: Sequence[Document] | None = None, **chunker_kwargs) -> CorpusIndex:
        """Build a CorpusIndex for a chunking strategy. The contextual chunker needs the LLM; it is
        injected here so callers just name the strategy."""
        if chunker_name == "contextual":
            chunker_kwargs.setdefault("llm", self.llm)
        chunker = build_chunker(chunker_name, **chunker_kwargs)
        pipeline = IngestionPipeline(chunker=chunker, embedder=self.embedder,
                                     vector_backend=self.config.store.vector_backend,
                                     tracer=self.tracer)
        return pipeline.ingest(list(documents) if documents is not None else self.corpus)
