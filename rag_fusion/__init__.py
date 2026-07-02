"""RAG-Fusion: LLM query broadening + parallel dense fan-out + Reciprocal Rank Fusion."""
from .config import Config
from .pipeline import Pipeline

__all__ = ["Config", "Pipeline"]
