"""Hybrid RAG — dense (embedding) and sparse (BM25) retrieval run in parallel over the same
chunks, fused into a single ranking (RRF by default, weighted score fusion by config).

Package contract: ``Config`` (frozen tunables) + ``Pipeline`` (retrieve / answer).
"""
from .config import Config
from .pipeline import Pipeline
from .retriever import HybridRetriever

__all__ = ["Config", "Pipeline", "HybridRetriever"]
