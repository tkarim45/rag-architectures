"""Leaf clustering — groups semantically similar documents so each group can be summarized into a
parent node. Agglomerative (bottom-up) clustering on the L2-normalized embeddings; n_clusters is
clamped below the leaf count so the fit never asks for more clusters than there are points."""
from __future__ import annotations

from sklearn.cluster import AgglomerativeClustering


def cluster(vectors, n_clusters):
    return AgglomerativeClustering(
        n_clusters=min(n_clusters, len(vectors) - 1)
    ).fit_predict(vectors)
