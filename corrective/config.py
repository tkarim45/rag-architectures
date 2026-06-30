"""Configuration for Corrective RAG (CRAG). Tunables live here so the pipeline reads like prod
code: change behavior by editing config, not the logic."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    chunker: str = "sentence"   # which offline index (built in common/index.py) to retrieve over
    min_relevant: int = 2       # graded-relevant chunks required before we trust the first retrieval
    top_k: int = 5              # chunks pulled from the index
    return_n: int = 5           # chunks handed to the generator
