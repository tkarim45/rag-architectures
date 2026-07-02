"""Configuration for the chunking-strategy architecture.

Every tunable lives here so the pipeline reads like production code: behavior changes by editing
config, never logic. The one deliberate *non*-tunable is the query path itself — this architecture
holds retrieval constant (plain dense search, identical to naive RAG) and varies only the
index-time chunking strategy, because that is the controlled experiment the package exists to run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core import ConfigurationError


@dataclass(frozen=True)
class Config:
    """Tunables for one chunking-strategy pipeline.

    Retrieval knobs (`top_k`, `final_k`, `max_context_chars`) apply uniformly to every strategy —
    keeping them identical across strategies is what makes the benchmark's numbers attributable to
    chunking alone. Strategy knobs (`sentence_window_size`, `fixed_*`) are forwarded to the core
    chunker constructors at index-build time via :meth:`chunker_kwargs`.
    """

    # ---- query path (held constant across strategies) -----------------------------------
    top_k: int = 8
    """Chunks pulled from the index per query. Deliberately > `final_k`: display-text dedup can
    collapse several hits into one passage (parent_child especially), so we over-fetch."""

    final_k: int = 5
    """Maximum passages handed to the generator after dedup + budgeting."""

    max_context_chars: int = 6000
    """Character budget for the assembled context block (~1.5k tokens). The knob that exposes
    parent_child's cost: whole-document display texts eat this budget fastest."""

    # ---- per-strategy index-build knobs --------------------------------------------------
    sentence_window_size: int = 1
    """`sentence_window` only: neighbors returned on each side of the matched sentence."""

    fixed_max_chars: int = 800
    """`fixed` only: character window size."""

    fixed_overlap_chars: int = 120
    """`fixed` only: overlap carried between consecutive windows."""

    # ---- observability -------------------------------------------------------------------
    diagnostics_matches: int = 3
    """Top hits whose (index_text, display_text) pair is recorded in retrieval diagnostics —
    the visible evidence of each strategy's match-small/return-big mechanics."""

    def __post_init__(self) -> None:
        if self.final_k < 1:
            raise ConfigurationError(f"final_k must be >= 1, got {self.final_k}")
        if self.top_k < self.final_k:
            raise ConfigurationError(
                f"top_k ({self.top_k}) must be >= final_k ({self.final_k}); the pipeline "
                "over-fetches because display-text dedup can collapse hits")
        if self.sentence_window_size < 0:
            raise ConfigurationError(
                f"sentence_window_size must be >= 0, got {self.sentence_window_size}")
        if self.fixed_max_chars < 1 or self.fixed_overlap_chars < 0:
            raise ConfigurationError("fixed_max_chars must be >= 1 and fixed_overlap_chars >= 0")
        if self.fixed_overlap_chars >= self.fixed_max_chars:
            raise ConfigurationError(
                f"fixed_overlap_chars ({self.fixed_overlap_chars}) must be < fixed_max_chars "
                f"({self.fixed_max_chars}) or windows never advance")
        if self.diagnostics_matches < 0:
            raise ConfigurationError(
                f"diagnostics_matches must be >= 0, got {self.diagnostics_matches}")

    def chunker_kwargs(self, strategy: str) -> dict[str, Any]:
        """Constructor kwargs for a core chunker, forwarded to `runtime.build_index`.

        Only strategies with tunable geometry get kwargs; the contextual chunker's LLM is injected
        by the runtime itself, and the rest are parameter-free.
        """
        if strategy == "sentence_window":
            return {"window": self.sentence_window_size}
        if strategy == "fixed":
            return {"max_chars": self.fixed_max_chars, "overlap_chars": self.fixed_overlap_chars}
        return {}
