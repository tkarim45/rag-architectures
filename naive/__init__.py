"""naive — the canonical dense-RAG baseline (embed query → top-k cosine → stuff → generate).

Every other architecture in this repo justifies its extra machinery against this package's
numbers. See README.md for the design rationale and ARCHITECTURE.md for the data flow.
"""
from .config import NaiveConfig
from .pipeline import Pipeline

#: Contract alias — the benchmark imports every package's tunables as ``Config``.
Config = NaiveConfig

__all__ = ["Config", "NaiveConfig", "Pipeline"]
