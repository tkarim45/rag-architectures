"""Tunables for two-stage retrieve-then-rerank.

Frozen dataclass: a config is an experiment identity — same Config + same index + same reranker
must reproduce the same ranking, so nothing mutates it after construction.

Why the defaults are what they are:

* ``candidate_k=20 ≫ final_k=5`` — the whole point of the funnel. Stage 1 (bi-encoder) is cheap
  and coarse, so it over-fetches for *recall*; stage 2 (cross-encoder) is expensive and sharp, so
  it runs only on those candidates for *precision*. The classic failure is setting ``candidate_k``
  too small: the reranker can only reorder what stage 1 surfaced — it can never recover a chunk
  stage 1 missed.
* ``use_sparse_candidates=False`` — dense-only stage 1 keeps the baseline clean. Flip it on to
  union BM25 candidates in when queries carry exact rare tokens (ids, names) that embeddings miss;
  it widens stage-1 recall at zero rerank-quality cost (the cross-encoder re-scores everything
  anyway).
* ``cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2"`` — the standard MS MARCO-trained
  passage reranker: strong precision at ~80 MB, fast enough on CPU.
* ``score_threshold=None`` — cross-encoder logits are model-specific and uncalibrated across
  models, so no universal cutoff exists. Set one only after inspecting your model's score
  distribution to drop confidently-irrelevant tails.
"""
from __future__ import annotations

from dataclasses import dataclass

from core import ConfigurationError


@dataclass(frozen=True)
class Config:
    """Rerank pipeline tunables. Validated at construction so misconfiguration fails loudly at
    pipeline build time, not at query time."""

    # ---- stage 1: candidate generation (recall) --------------------------------------------
    candidate_k: int = 20                   #: candidates fetched per retrieval branch
    use_sparse_candidates: bool = False     #: also union BM25 candidates into stage 1

    # ---- stage 2: reranking (precision) -----------------------------------------------------
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  #: HF cross-encoder name
    batch_size: int = 32                    #: (query, passage) pairs scored per forward pass
    score_threshold: float | None = None    #: drop reranked chunks scoring below this (optional)

    # ---- output ------------------------------------------------------------------------------
    final_k: int = 5                        #: reranked chunks kept for the retrieval result

    # ---- offline index -------------------------------------------------------------------
    chunker: str = "sentence"               #: chunking strategy when the pipeline builds its own index

    # ---- context assembly ----------------------------------------------------------------
    max_context_passages: int = 5           #: passages handed to the generator
    max_context_chars: int = 6000           #: character budget (≈ tokens × 4) for the context block

    def __post_init__(self) -> None:
        if self.candidate_k < 1:
            raise ConfigurationError("candidate_k must be >= 1")
        if self.final_k < 1:
            raise ConfigurationError("final_k must be >= 1")
        if self.final_k > self.candidate_k:
            raise ConfigurationError(
                f"final_k ({self.final_k}) > candidate_k ({self.candidate_k}): the funnel must "
                "narrow — stage 2 cannot return more than stage 1 surfaced")
        if self.batch_size < 1:
            raise ConfigurationError("batch_size must be >= 1")
        if not self.cross_encoder_model:
            raise ConfigurationError("cross_encoder_model must be a non-empty model name")
        if self.max_context_passages < 1 or self.max_context_chars < 1:
            raise ConfigurationError("context budget must be positive")
