"""Rerank RAG — two-stage retrieve-then-rerank: a cheap high-recall candidate stage (dense,
optionally + BM25 union) followed by a cross-encoder precision reranker over the candidates.

Package contract: ``Config`` (frozen tunables) + ``Pipeline`` (retrieve / answer). The reranker is
injectable (``Reranker`` protocol) so offline tests run ``LexicalOverlapReranker`` without torch.
"""
from .config import Config
from .pipeline import Pipeline
from .reranker import CrossEncoderReranker, LexicalOverlapReranker, Reranker
from .retriever import RerankRetriever

__all__ = ["Config", "Pipeline", "Reranker", "CrossEncoderReranker", "LexicalOverlapReranker",
           "RerankRetriever"]
