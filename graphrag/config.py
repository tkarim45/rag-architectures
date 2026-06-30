"""Configuration for GraphRAG. Tunables live here so the pipeline reads like prod code: change
behavior by editing config, not the logic."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    top_k: int = 5   # docs handed to the generator after traversal
    hops: int = 2    # how many shared-entity edges to expand out from the seed docs
