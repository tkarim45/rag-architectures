"""Chunking strategies.

Every chunker maps documents → chunks, controlling two independent things: the *index text* (what
gets embedded / BM25-indexed — precision) and the *display text* (what the generator reads —
context). The registry makes strategies addressable by name from config and from the chunking
architecture package, which benchmarks them head-to-head.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from ..llm.base import LLM
from ..types import Chunk, Document

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


class Chunker(ABC):
    """Base chunker: subclasses implement per-document chunking; the base handles iteration and
    stable id assignment."""

    name: str = "base"

    def chunk(self, documents: Sequence[Document]) -> list[Chunk]:
        out: list[Chunk] = []
        for doc in documents:
            out.extend(self.chunk_document(doc))
        return out

    @abstractmethod
    def chunk_document(self, doc: Document) -> Iterable[Chunk]: ...


class WholeDocumentChunker(Chunker):
    """One chunk per document — the degenerate baseline every strategy is compared against."""

    name = "whole"

    def chunk_document(self, doc: Document) -> Iterable[Chunk]:
        yield Chunk(doc.doc_id, doc.doc_id, doc.text, doc.text,
                    metadata={"strategy": self.name})


class SentenceChunker(Chunker):
    """One chunk per sentence: maximal retrieval precision, minimal generation context."""

    name = "sentence"

    def chunk_document(self, doc: Document) -> Iterable[Chunk]:
        for i, sentence in enumerate(split_sentences(doc.text)):
            yield Chunk(f"{doc.doc_id}::s{i}", doc.doc_id, sentence, sentence,
                        metadata={"strategy": self.name, "position": i})


@dataclass
class FixedSizeChunker(Chunker):
    """Character-window chunking with overlap — the industry default for long documents. Splits on
    sentence boundaries where possible so windows don't cut mid-sentence."""

    max_chars: int = 800
    overlap_chars: int = 120
    name = "fixed"

    def chunk_document(self, doc: Document) -> Iterable[Chunk]:
        sentences = split_sentences(doc.text) or [doc.text]
        window: list[str] = []
        length = 0
        index = 0
        for sentence in sentences:
            if window and length + len(sentence) > self.max_chars:
                text = " ".join(window)
                yield Chunk(f"{doc.doc_id}::f{index}", doc.doc_id, text, text,
                            metadata={"strategy": self.name, "position": index})
                index += 1
                # keep a tail of sentences as overlap for the next window
                kept: list[str] = []
                kept_len = 0
                for prev in reversed(window):
                    if kept_len + len(prev) > self.overlap_chars:
                        break
                    kept.insert(0, prev)
                    kept_len += len(prev)
                window, length = kept, kept_len
            window.append(sentence)
            length += len(sentence)
        if window:
            text = " ".join(window)
            yield Chunk(f"{doc.doc_id}::f{index}", doc.doc_id, text, text,
                        metadata={"strategy": self.name, "position": index})


@dataclass
class SentenceWindowChunker(Chunker):
    """Index a single sentence; return it with ±`window` neighbors. Match small, read large."""

    window: int = 1
    name = "sentence_window"

    def chunk_document(self, doc: Document) -> Iterable[Chunk]:
        sentences = split_sentences(doc.text)
        for i, sentence in enumerate(sentences):
            lo, hi = max(0, i - self.window), min(len(sentences), i + self.window + 1)
            yield Chunk(f"{doc.doc_id}::w{i}", doc.doc_id, sentence, " ".join(sentences[lo:hi]),
                        metadata={"strategy": self.name, "position": i})


class ParentChildChunker(Chunker):
    """Index sentences (children), return the full parent document — the small-to-big pattern."""

    name = "parent_child"

    def chunk_document(self, doc: Document) -> Iterable[Chunk]:
        for i, sentence in enumerate(split_sentences(doc.text)):
            yield Chunk(f"{doc.doc_id}::p{i}", doc.doc_id, sentence, doc.text,
                        metadata={"strategy": self.name, "position": i, "parent": doc.doc_id})


CONTEXT_PROMPT = (
    "Document:\n{document}\n\nIn one short sentence, give the context a reader needs to situate "
    "any excerpt from this document (who/what it is about). Reply with only the sentence.")


@dataclass
class ContextualChunker(Chunker):
    """Anthropic-style contextual retrieval: prepend an LLM-written one-line document context to
    each sentence's index text so isolated chunks stop being ambiguous ('the company' → which?).
    One LLM call per document at build time; pair with the LLM cache for cheap rebuilds."""

    llm: LLM = None  # type: ignore[assignment]
    name = "contextual"

    def chunk_document(self, doc: Document) -> Iterable[Chunk]:
        from ..llm.base import CompletionRequest

        context = self.llm.complete(CompletionRequest(
            prompt=CONTEXT_PROMPT.format(document=doc.text), max_tokens=60)).text.strip()
        for i, sentence in enumerate(split_sentences(doc.text)):
            yield Chunk(f"{doc.doc_id}::c{i}", doc.doc_id, f"{context} {sentence}", sentence,
                        metadata={"strategy": self.name, "position": i, "context": context})


ChunkerFactory = Callable[..., Chunker]

CHUNKER_REGISTRY: dict[str, ChunkerFactory] = {
    "whole": WholeDocumentChunker,
    "sentence": SentenceChunker,
    "fixed": FixedSizeChunker,
    "sentence_window": SentenceWindowChunker,
    "parent_child": ParentChildChunker,
    "contextual": ContextualChunker,
}


def build_chunker(name: str, **kwargs) -> Chunker:
    try:
        factory = CHUNKER_REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown chunker {name!r}; known: {sorted(CHUNKER_REGISTRY)}") from None
    return factory(**kwargs)
