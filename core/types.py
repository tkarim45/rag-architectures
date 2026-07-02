"""Core domain models shared by every architecture package.

These are the *only* shapes that cross package boundaries. Retrievers consume `Query` and produce
`RetrievalResult`; generators consume `ContextBlock` and produce `GeneratedAnswer`; pipelines wrap
the whole exchange in a `PipelineResult` that carries diagnostics and trace spans for observability.

Everything is a frozen dataclass where identity matters (documents, chunks) and a mutable dataclass
where accumulation matters (results). No framework types (numpy arrays, store handles) leak through
here — that keeps architecture packages decoupled from backend choices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


# --------------------------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Document:
    """A source document as ingested — the unit of provenance and of retrieval scoring."""

    doc_id: str
    title: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """An indexable unit derived from a document.

    `index_text` is what gets embedded / keyword-indexed; `display_text` is what the generator
    reads on a hit. Separating the two is the seam that sentence-window, parent-child and
    contextual chunking exploit: match on something small and precise, return something larger
    and richer.
    """

    chunk_id: str
    doc_id: str
    index_text: str
    display_text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Query:
    """A retrieval request. `variants` holds rewrites/expansions produced upstream (multi-query,
    HyDE hypotheses, corrective rewrites) so downstream stages can log what was actually searched."""

    text: str
    top_k: int = 5
    variants: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredChunk:
    """One retrieval hit: a chunk, the score that ranked it, and which retriever produced it."""

    chunk: Chunk
    score: float
    retriever: str = ""

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id


@dataclass
class RetrievalResult:
    """Ranked output of a retriever (or a whole retrieval graph).

    `diagnostics` is the architecture's story of *how* it retrieved — routes taken, queries
    generated, grades assigned, hops walked. Benchmarks and traces read it; generators don't.
    """

    query: Query
    chunks: list[ScoredChunk] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_ids(self) -> list[str]:
        return [s.chunk_id for s in self.chunks]

    @property
    def doc_ids(self) -> list[str]:
        """Ranked unique doc ids — first occurrence wins. This is what retrieval metrics score."""
        seen: set[str] = set()
        out: list[str] = []
        for s in self.chunks:
            if s.doc_id not in seen:
                seen.add(s.doc_id)
                out.append(s.doc_id)
        return out

    def top(self, k: int) -> list[ScoredChunk]:
        return self.chunks[:k]


# --------------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextBlock:
    """Assembled generation context: the final prompt-ready text plus provenance of every
    passage that made it in (for citations and for debugging truncation)."""

    text: str
    chunk_ids: tuple[str, ...] = ()
    doc_ids: tuple[str, ...] = ()
    truncated: bool = False


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(self.input_tokens + other.input_tokens,
                          self.output_tokens + other.output_tokens)


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    citations: tuple[str, ...] = ()          # doc ids the answer is grounded in
    usage: TokenUsage = TokenUsage()
    abstained: bool = False                  # generator said "I don't know"


# --------------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """End-to-end record of one query through one architecture. The benchmark consumes
    (`doc_ids`, `context`, `diagnostics`); interactive callers consume `answer`."""

    query: Query
    retrieval: RetrievalResult
    context: ContextBlock
    answer: GeneratedAnswer | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_ids(self) -> list[str]:
        return self.retrieval.doc_ids
