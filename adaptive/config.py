"""Configuration for Adaptive RAG (router). Tunables live here so the pipeline reads like prod code:
change behavior by editing config, not the logic."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    chunker: str = "sentence"   # which offline index (built in common/index.py) the dense/broad paths use
    top_k: int = 5              # chunks/docs handed to the generator on the routed path
