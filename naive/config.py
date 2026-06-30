"""Configuration for Naive RAG. Tunables live here so the pipeline reads like prod code: change
behavior by editing config, not the logic."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    chunker: str = "sentence"   # which offline index (built in common/index.py) to retrieve over
    top_k: int = 8              # chunks pulled from the index
    return_n: int = 5           # chunks handed to the generator
