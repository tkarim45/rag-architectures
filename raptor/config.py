"""RAPTOR tunables.

Every knob the tree builder and the collapsed-tree retriever read lives here — the modules
themselves contain no magic numbers. Defaults are sized for this repo's small fictional corpus
(~14 short documents); the tuning table in `ARCHITECTURE.md` explains how each knob moves
behavior on larger corpora.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Frozen so a benchmark run's configuration is immutable, hashable, and safely shareable
    between the offline builder (`build_tree`) and every online `Pipeline` that reads the tree.
    """

    # ---- offline: tree construction ----------------------------------------------------

    max_levels: int = 4
    """Maximum number of summary levels above the leaves. Recursion also stops early when a
    level collapses to a single root node, or when clustering stops reducing the node count."""

    max_clusters: int = 6
    """Upper bound of the BIC model-selection sweep: n_components ranges over
    1..min(max_clusters, n_nodes). Sarthi et al. select k by BIC rather than fixing it."""

    soft_threshold: float = 0.1
    """Posterior-probability cutoff for *soft* cluster membership. A node joins every Gaussian
    component whose responsibility for it exceeds this value, so one document can feed multiple
    cluster summaries — the paper's mechanism for topics that straddle clusters."""

    min_cluster_size: int = 2
    """Clusters with fewer members than this are dissolved and their members reassigned to their
    next-best surviving cluster. Summarizing a singleton just paraphrases one node — it adds an
    LLM call and a near-duplicate embedding without abstracting anything."""

    covariance_type: str = "spherical"
    """GaussianMixture covariance structure. `spherical` keeps the parameter count O(k·d) — with
    tiny-n / high-d inputs (we skip the paper's UMAP reduction; see `clustering.py`) a `full`
    covariance would be singular and BIC's parameter penalty would drown the likelihood."""

    random_state: int = 13
    """Seed for GaussianMixture initialization: identical corpora build identical trees, which
    the benchmark relies on when it builds the tree once and shares it."""

    summary_max_tokens: int = 256
    """Completion budget per cluster summary. Summaries should compress, not re-narrate."""

    # ---- online: collapsed-tree retrieval ----------------------------------------------

    top_nodes: int = 8
    """Hard cap on selected nodes per query, applied on top of the token budget."""

    max_context_tokens: int = 1500
    """Token budget for greedy node selection (approximated as chars/4, matching the paper's
    collapsed-tree procedure of adding nodes by similarity until a token threshold)."""

    # ---- online: context assembly -------------------------------------------------------

    max_context_passages: int = 8
    """Passage cap handed to core.ContextBuilder. Node texts repeated across per-doc synthetic
    chunks are deduplicated by the builder, so this counts *unique* node texts."""

    answer_max_tokens: int = 300
    """Completion budget for the final answer generator."""
