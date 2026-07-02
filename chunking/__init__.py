"""Chunking-strategy RAG — the architecture where the *index*, not the query path, is the variable.

The same dense retriever over the same corpus swings materially with the chunking strategy used at
build time. This package benchmarks the three strategies that decouple what is matched from what
is returned (`STRATEGIES`), profiles all six core strategies (`STRATEGY_PROFILES`), and holds the
online query path constant so every score delta is attributable to chunking alone.
"""
from .config import Config
from .pipeline import Pipeline
from .strategies import STRATEGIES, STRATEGY_PROFILES, StrategyProfile, profile

__all__ = ["Config", "Pipeline", "STRATEGIES", "STRATEGY_PROFILES", "StrategyProfile", "profile"]
