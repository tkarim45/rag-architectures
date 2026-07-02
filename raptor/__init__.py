"""RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval.

Implements Sarthi et al. 2024 (arXiv:2401.18059) on the core framework: an offline tree of
recursively clustered-and-summarized document embeddings, queried online with the paper's
better-performing collapsed-tree strategy. See `README.md` for results and `ARCHITECTURE.md`
for the data flow, failure modes, and tuning guide.
"""
from .config import Config
from .pipeline import Pipeline
from .tree import RaptorTree, build_tree

__all__ = ["Config", "Pipeline", "RaptorTree", "build_tree"]
