"""Soft Gaussian-mixture clustering for one tree level.

Faithful to Sarthi et al. (2024) in the two decisions that matter:

* **model selection by BIC** — the number of Gaussian components is not a fixed hyperparameter;
  we sweep n_components over 1..min(max_clusters, n_nodes) and keep the BIC-optimal mixture.
* **soft assignment** — a node joins *every* component whose posterior responsibility for it
  exceeds `config.soft_threshold` (default 0.1), so a document that straddles topics feeds
  multiple cluster summaries instead of being forced into exactly one.

Deliberate departure from the paper: **no UMAP dimensionality reduction.** The paper runs UMAP
before the GMM because it clusters thousands of chunk embeddings; at this corpus's scale (tens of
nodes) UMAP would add a heavyweight dependency and behave erratically (its neighborhood graph is
degenerate for tiny n). We fit the GMM on the raw embeddings instead and compensate with a
`spherical` covariance so the parameter count stays O(k·d) — an honest small-corpus trade-off,
recorded here rather than hidden.
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

from .config import Config


def cluster_level(embeddings: np.ndarray, config: Config) -> list[tuple[int, ...]]:
    """Softly cluster one level's node embeddings.

    Args:
        embeddings: (n, dim) float32, L2-normalized rows — one row per node at this level.
        config: RAPTOR tunables (BIC sweep range, soft threshold, min cluster size, seed).

    Returns:
        Clusters as tuples of row indices, largest first. Every input index appears in at least
        one cluster; an index may appear in several (soft assignment). Guarantees at least one
        cluster for non-empty input.
    """
    n_nodes = int(embeddings.shape[0])
    if n_nodes == 0:
        return []
    # Degenerate case: with <= 2 nodes a mixture model is meaningless (BIC would compare k=1 vs
    # k=n on almost no evidence) — group everything into the single cluster it obviously is.
    if n_nodes <= 2:
        return [tuple(range(n_nodes))]

    gmm = _fit_best_gmm(embeddings, config)
    responsibilities = gmm.predict_proba(embeddings)          # (n, k) posteriors
    clusters = _soft_assign(responsibilities, config.soft_threshold)
    clusters = _enforce_min_size(clusters, responsibilities, config.min_cluster_size)
    return sorted(clusters, key=len, reverse=True)


def _fit_best_gmm(embeddings: np.ndarray, config: Config) -> GaussianMixture:
    """Sweep n_components and keep the BIC-optimal mixture (lower BIC = better)."""
    n_nodes = int(embeddings.shape[0])
    max_k = max(1, min(config.max_clusters, n_nodes))
    best: GaussianMixture | None = None
    best_bic = np.inf
    for k in range(1, max_k + 1):
        candidate = GaussianMixture(
            n_components=k,
            covariance_type=config.covariance_type,
            random_state=config.random_state,
            n_init=1,
        ).fit(embeddings)
        bic = float(candidate.bic(embeddings))
        if bic < best_bic:
            best, best_bic = candidate, bic
    assert best is not None  # loop runs at least once (max_k >= 1)
    return best


def _soft_assign(responsibilities: np.ndarray, threshold: float) -> list[list[int]]:
    """Node i joins every component c with P(c|i) > threshold; its argmax component is always
    included so no node is orphaned when its posterior mass is spread thin."""
    n_nodes, n_components = responsibilities.shape
    members: list[list[int]] = [[] for _ in range(n_components)]
    for i in range(n_nodes):
        assigned = {int(c) for c in np.flatnonzero(responsibilities[i] > threshold)}
        assigned.add(int(np.argmax(responsibilities[i])))
        for c in sorted(assigned):
            members[c].append(i)
    return [m for m in members if m]


def _enforce_min_size(clusters: list[list[int]], responsibilities: np.ndarray,
                      min_size: int) -> list[tuple[int, ...]]:
    """Dissolve clusters below `min_size`, reassigning members to their best surviving cluster.

    Summarizing a singleton cluster is pure waste — the LLM paraphrases one node and the tree
    gains a near-duplicate. If *every* cluster is undersized (tiny levels), keep the largest one
    and fold everything into it rather than returning nothing.
    """
    survivors = [c for c in clusters if len(c) >= min_size]
    if not survivors:
        merged = sorted({i for c in clusters for i in c})
        return [tuple(merged)]

    # Component index of each surviving cluster is lost after filtering, so reassign dissolved
    # members by posterior over the surviving clusters' member-mean responsibility columns:
    # simplest correct proxy — pick the surviving cluster whose members the node is most
    # responsible to share a component with (argmax over original posteriors restricted to
    # surviving clusters' dominant components).
    surviving_sets = [set(c) for c in survivors]
    dissolved = [i for c in clusters if len(c) < min_size for i in c]
    for i in dissolved:
        if any(i in s for s in surviving_sets):
            continue  # soft assignment already put it in a surviving cluster too
        # Score each surviving cluster by mean posterior similarity of node i to its members'
        # argmax components; fall back to the largest cluster on ties.
        best_idx = 0
        best_score = -1.0
        for idx, members in enumerate(survivors):
            comps = [int(np.argmax(responsibilities[j])) for j in members]
            score = float(np.mean([responsibilities[i, c] for c in comps]))
            if score > best_score:
                best_idx, best_score = idx, score
        surviving_sets[best_idx].add(i)
    return [tuple(sorted(s)) for s in surviving_sets]
