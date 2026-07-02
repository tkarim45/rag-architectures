"""Multi-query RAG: LLM query expansion + parallel dense fan-out + round-robin merge."""
from .config import Config
from .pipeline import Pipeline

__all__ = ["Config", "Pipeline"]
