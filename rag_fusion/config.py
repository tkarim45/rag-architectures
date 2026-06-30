"""Configuration for RAG-Fusion."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    chunker: str = "sentence"   # which offline index (built in common/index.py) to retrieve over
    n_queries: int = 3          # LLM rephrasings generated per query (in addition to the original)
    top_k: int = 8              # chunks kept from the RRF-fused rankings
    return_n: int = 5           # chunks handed to the generator
