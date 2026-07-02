"""Tunables for hybrid (dense + BM25) retrieval.

Every knob that changes retrieval behavior lives in this frozen dataclass — the pipeline and
retriever contain no magic numbers. Frozen because a config is an *experiment identity*: two runs
with the same Config over the same index must produce the same ranking, so nothing may mutate it
mid-flight.

Why the defaults are what they are:

* ``fusion="rrf"`` — BM25 scores are unbounded term-frequency sums and cosine similarities live in
  [-1, 1]; the two are on incomparable scales, and any direct score arithmetic silently lets one
  branch dominate. RRF (Cormack et al. 2009) fuses on *rank position only*, which makes it
  scale-free and hyperparameter-light — the safe default. Switch to ``"weighted"`` only when you
  have calibrated scores and validation data to tune ``weights`` against.
* ``rrf_k=60`` — the canonical constant from the paper; it damps the head of each ranking so a
  single branch's #1 cannot single-handedly outvote broad agreement lower down.
* ``dense_k=sparse_k=12 > final_k=8`` — each branch over-fetches so fusion has genuine overlap to
  reward; if branch k equalled final k, fusion would mostly interleave two disjoint lists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core import ConfigurationError

FusionMethod = Literal["rrf", "weighted"]
Normalization = Literal["minmax", "zscore"]


@dataclass(frozen=True)
class Config:
    """Hybrid retrieval tunables. Validated at construction so misconfiguration fails loudly at
    pipeline build time, not silently at query time."""

    # ---- branch fan-out ------------------------------------------------------------------
    dense_k: int = 12                       #: candidates pulled from the vector index
    sparse_k: int = 12                      #: candidates pulled from BM25

    # ---- fusion --------------------------------------------------------------------------
    fusion: FusionMethod = "rrf"            #: "rrf" (rank-based, safe) | "weighted" (score-based)
    rrf_k: int = 60                         #: RRF damping constant (canonical value: 60)
    weights: tuple[float, float] = (0.5, 0.5)   #: (dense, sparse) weights for weighted fusion
    normalization: Normalization = "minmax"     #: score normalization for weighted fusion

    # ---- output --------------------------------------------------------------------------
    final_k: int = 8                        #: fused chunks kept for the retrieval result

    # ---- offline index -------------------------------------------------------------------
    chunker: str = "sentence"               #: chunking strategy when the pipeline builds its own index

    # ---- context assembly ----------------------------------------------------------------
    max_context_passages: int = 5           #: passages handed to the generator
    max_context_chars: int = 6000           #: character budget (≈ tokens × 4) for the context block

    def __post_init__(self) -> None:
        if self.dense_k < 1 or self.sparse_k < 1:
            raise ConfigurationError("dense_k and sparse_k must be >= 1")
        if self.final_k < 1:
            raise ConfigurationError("final_k must be >= 1")
        if self.fusion not in ("rrf", "weighted"):
            raise ConfigurationError(f"unknown fusion method: {self.fusion!r}")
        if self.fusion == "rrf" and self.rrf_k < 1:
            raise ConfigurationError("rrf_k must be >= 1")
        if self.fusion == "weighted":
            if len(self.weights) != 2:
                raise ConfigurationError("weights must be (dense_weight, sparse_weight)")
            if any(w < 0 for w in self.weights) or sum(self.weights) <= 0:
                raise ConfigurationError("weights must be non-negative and sum to > 0")
            if self.normalization not in ("minmax", "zscore"):
                raise ConfigurationError(f"unknown normalization: {self.normalization!r}")
        if self.max_context_passages < 1 or self.max_context_chars < 1:
            raise ConfigurationError("context budget must be positive")
