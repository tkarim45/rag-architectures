"""Configuration for HyDE (Hypothetical Document Embeddings) RAG."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    chunker: str = "sentence"   # which offline index (built in common/index.py) to retrieve over
    top_k: int = 8              # chunks pulled from the index
    return_n: int = 5           # chunks handed to the generator
    hyde_max_tokens: int = 160  # budget for the drafted hypothetical answer
