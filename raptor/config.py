"""Configuration for RAPTOR. Tunables live here so the pipeline reads like prod code: change
behavior by editing config, not the logic."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    n_clusters: int = 4   # target number of leaf clusters to summarize into parent nodes
    top_k: int = 5        # docs handed to the generator after expanding the top-scoring nodes
