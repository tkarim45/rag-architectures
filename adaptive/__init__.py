"""adaptive — Adaptive-RAG: route each query, by predicted complexity, to the cheapest
sufficient retrieval strategy (Jeong et al. 2024, arXiv:2403.14403).

An LLM classifier labels each question A (no retrieval), B (single-step dense) or
C (multi-step iterative fused retrieval); only the chosen route runs. See README.md for the
bet and the risk, ARCHITECTURE.md for the data flow.
"""
from .config import AdaptiveConfig
from .pipeline import Pipeline

#: Contract alias — the benchmark imports every package's tunables as ``Config``.
Config = AdaptiveConfig

__all__ = ["Config", "AdaptiveConfig", "Pipeline"]
