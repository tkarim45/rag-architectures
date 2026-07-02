"""sparse — BM25 lexical retrieval done properly (explicit analyzer, tunable k1/b).

The lexical counterpart to the ``naive`` dense baseline: exact-term matching over an inverted
index. See README.md for when lexical wins (rare entities, IDs) and where it fails (vocabulary
mismatch), and ARCHITECTURE.md for the data flow.
"""
from .config import SparseConfig
from .pipeline import Pipeline

#: Contract alias — the benchmark imports every package's tunables as ``Config``.
Config = SparseConfig

__all__ = ["Config", "SparseConfig", "Pipeline"]
