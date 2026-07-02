"""Tunables for the agentic (ReAct) retrieval pipeline.

Every knob that changes agent behavior lives in this frozen dataclass — the agent loop, tool layer
and evidence ranking contain no magic numbers. Frozen because a config is an *experiment identity*:
two runs with the same Config over the same index must differ only by LLM stochasticity, never by
mutated settings.

Why the defaults are what they are:

* ``max_steps=8`` — the corpus questions need at most 3 hops; 8 gives the agent room to recover
  from one or two dead-end searches while still bounding worst-case cost at 8 LLM calls + tools.
  ReAct trajectories that haven't converged by ~2× the expected hop depth almost never converge
  (Yao et al. 2023 report the same plateau) — past that, budget cutoff is cheaper than hope.
* ``search_k=keyword_k=5`` — per-probe fan-out. Small on purpose: the agent compensates for a
  narrow k with *follow-up* searches, which is the whole point of the architecture. A large k
  would flood the scratchpad and the evidence log with weak hits.
* ``final_k=8 > search_k`` — the evidence log accumulates across steps, so the final ranking pool
  is bigger than any single probe; 8 keeps recall headroom for the benchmark's recall@5.
* ``max_tool_output_chars=2000`` — observations feed straight back into the next-step prompt, so
  unbounded tool output makes the loop quadratically expensive. 2000 chars ≈ 500 tokens holds a
  full corpus document; raise it for corpora with longer documents (see ARCHITECTURE.md failure
  modes: truncation can drop the key fact).
* ``recency_weight=0.9`` — evidence ranking blends touch frequency with trajectory recency. Kept
  strictly < 1 so one extra touch always outranks any recency bonus; see ``evidence.py`` for the
  full rationale.
"""
from __future__ import annotations

from dataclasses import dataclass

from core import ConfigurationError


@dataclass(frozen=True)
class Config:
    """Agentic retrieval tunables. Validated at construction so misconfiguration fails loudly at
    pipeline build time, not silently mid-trajectory."""

    # ---- agent loop budgets --------------------------------------------------------------
    max_steps: int = 8                  #: hard cap on ReAct steps (each step = one LLM call)
    max_decision_tokens: int = 512      #: token budget for one {thought, action} decision
    max_tool_output_chars: int = 2000   #: observations are truncated to this before the scratchpad

    # ---- per-tool fan-out ----------------------------------------------------------------
    search_k: int = 5                   #: hits returned by the dense `search` tool
    keyword_k: int = 5                  #: hits returned by the BM25 `keyword_search` tool
    snippet_chars: int = 240            #: snippet length per hit in search-tool observations

    # ---- evidence ranking ----------------------------------------------------------------
    final_k: int = 8                    #: evidence chunks kept for the RetrievalResult
    recency_weight: float = 0.9         #: recency bonus weight in the evidence blend (must be < 1)

    # ---- trajectory cache ----------------------------------------------------------------
    cache_size: int = 16                #: questions whose trajectories are memoized (0 disables)

    # ---- offline index -------------------------------------------------------------------
    chunker: str = "sentence"           #: chunking strategy when the pipeline builds its own index

    # ---- context assembly ----------------------------------------------------------------
    max_context_passages: int = 5       #: passages handed to the fallback generator
    max_context_chars: int = 6000       #: character budget (≈ tokens × 4) for the context block

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ConfigurationError("max_steps must be >= 1")
        if self.max_decision_tokens < 1:
            raise ConfigurationError("max_decision_tokens must be >= 1")
        if self.max_tool_output_chars < 1:
            raise ConfigurationError("max_tool_output_chars must be >= 1")
        if self.search_k < 1 or self.keyword_k < 1:
            raise ConfigurationError("search_k and keyword_k must be >= 1")
        if self.snippet_chars < 1:
            raise ConfigurationError("snippet_chars must be >= 1")
        if self.final_k < 1:
            raise ConfigurationError("final_k must be >= 1")
        if not 0.0 <= self.recency_weight < 1.0:
            raise ConfigurationError(
                "recency_weight must be in [0, 1) so touch frequency always dominates recency")
        if self.cache_size < 0:
            raise ConfigurationError("cache_size must be >= 0")
        if self.max_context_passages < 1 or self.max_context_chars < 1:
            raise ConfigurationError("context budget must be positive")
