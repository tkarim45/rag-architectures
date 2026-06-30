"""Configuration for Sparse (BM25) RAG."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    chunker: str = "sentence"
    top_k: int = 8
    return_n: int = 5
