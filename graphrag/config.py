"""GraphRAG tunables.

Every knob the pipeline reads lives here, so behavior changes by editing config — never by editing
logic. Defaults are scaled honestly to the shared 14-document corpus: two hops covers most of the
dataset's relation chains, five docs of context fits the generator budget, and three communities is
plenty when Louvain finds only a handful of clusters at this scale.
"""
from __future__ import annotations

from dataclasses import dataclass

from core import ConfigurationError

#: Search modes. "local" is entity-centric neighborhood traversal, "global" is community
#: map-reduce, "auto" asks the LLM to pick per question.
MODES: tuple[str, ...] = ("local", "global", "auto")


@dataclass(frozen=True)
class Config:
    # ---- online: routing -----------------------------------------------------------------
    mode: str = "local"
    """Which search to run: "local" | "global" | "auto" (cheap LLM classification per query)."""

    # ---- online: local search ------------------------------------------------------------
    max_hops: int = 2
    """How far to expand the entity neighborhood from the seed entities. Hop 0 is the seeds
    themselves; each extra hop trades precision for the chance to reach a bridge document."""

    top_k_docs: int = 5
    """How many ranked provenance documents the search hands to the context builder."""

    entity_match_min_chars: int = 4
    """Lexical fallback threshold: a graph entity only counts as 'mentioned in the question' if
    its normalized name is at least this long. Guards against short names ("orsa" is fine at 4,
    a hypothetical entity "ai" would match half of all questions)."""

    # ---- online: global search -------------------------------------------------------------
    max_communities: int = 3
    """Reduce step keeps at most this many top-rated communities."""

    min_community_rating: int = 1
    """Communities rated below this (0-10 scale) are dropped even if slots remain."""

    # ---- offline: graph build --------------------------------------------------------------
    min_community_size: int = 2
    """Louvain communities smaller than this are not summarized — a singleton entity carries no
    relational structure worth an LLM call."""

    louvain_seed: int = 42
    """Seed for Louvain community detection so offline builds are reproducible."""

    extraction_max_tokens: int = 1024
    """Token budget for the per-document entity/relation extraction call."""

    summary_max_tokens: int = 256
    """Token budget for each community summary."""

    # ---- context assembly --------------------------------------------------------------------
    max_context_passages: int = 5
    """Passages the ContextBuilder keeps (one whole-doc chunk per provenance doc here)."""

    max_context_chars: int = 6000
    """Character budget for the assembled context block."""

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ConfigurationError(f"graphrag mode must be one of {MODES}, got {self.mode!r}")
        if self.max_hops < 0:
            raise ConfigurationError(f"max_hops must be >= 0, got {self.max_hops}")
        if self.top_k_docs < 1:
            raise ConfigurationError(f"top_k_docs must be >= 1, got {self.top_k_docs}")
        if self.max_communities < 1:
            raise ConfigurationError(f"max_communities must be >= 1, got {self.max_communities}")
