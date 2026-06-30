"""Configuration for Chunking-strategy RAG. Same dense retriever throughout — the only knob that
matters here is *which* chunk index (granularity) the retriever reads over."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    chunker: str = "parent_child"   # offline index granularity: sentence_window / parent_child / contextual
    top_k: int = 8                  # chunks pulled from the index
    return_n: int = 5               # return_texts handed to the generator
