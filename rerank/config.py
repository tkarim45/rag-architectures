"""Configuration for Rerank RAG (cross-encoder). Tunables live here so the pipeline reads like prod
code: change behavior by editing config, not the logic."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    chunker: str = "sentence"                                  # which offline index to retrieve over
    recall_k: int = 15                                         # min dense candidates handed to reranker
    top_k: int = 8                                             # chunks kept after cross-encoder rerank
    return_n: int = 5                                          # chunks handed to the generator
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"        # cross-encoder used for rescoring
