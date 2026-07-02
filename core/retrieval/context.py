"""Context assembly: turn a ranked chunk list into the text block a generator reads.

Owns the three decisions every RAG system makes at this seam:
  * dedup — identical display texts collapse (parent-child chunking yields many hits → one doc)
  * budget — a character budget (≈ tokens × 4) truncates the tail, recorded via `truncated`
  * provenance — each passage is numbered and the block remembers which chunks/docs made it in,
    so answers can cite and benchmarks can audit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..types import ContextBlock, ScoredChunk


@dataclass(frozen=True)
class ContextBuilder:
    max_passages: int = 5
    max_chars: int = 6000

    def build(self, chunks: Sequence[ScoredChunk]) -> ContextBlock:
        seen_texts: set[str] = set()
        passages: list[str] = []
        chunk_ids: list[str] = []
        doc_ids: list[str] = []
        used = 0
        truncated = False
        for hit in chunks:
            text = hit.chunk.display_text
            if text in seen_texts:
                continue
            if len(passages) >= self.max_passages or used + len(text) > self.max_chars:
                truncated = True
                break
            seen_texts.add(text)
            passages.append(f"[{len(passages) + 1}] {text}")
            chunk_ids.append(hit.chunk_id)
            if hit.doc_id not in doc_ids:
                doc_ids.append(hit.doc_id)
            used += len(text)
        return ContextBlock(text="\n\n".join(passages), chunk_ids=tuple(chunk_ids),
                            doc_ids=tuple(doc_ids), truncated=truncated)
