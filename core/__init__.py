"""rag-architectures core framework.

Layering (imports only point downward):

    types / errors                          domain models, error taxonomy
    config / telemetry                      env-layered settings, spans + structured logs
    llm / embeddings / stores               provider backends behind protocols (+ caches, fakes)
    ingestion / retrieval / generation      chunkers → CorpusIndex; fusion + context; generator
    evaluation / dataset                    metrics + judge; the fictional labeled corpus
    runtime                                 composition root (from_env / for_testing)

Architecture packages import from here and only from here.
"""
from .config import CoreConfig
from .dataset import LabeledQuestion, documents, questions
from .errors import (ConfigurationError, ProviderError, RagArchError, RateLimitError,
                     RetrievalError, StructuredOutputError)
from .evaluation import CorrectnessJudge, hit_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k
from .generation import AnswerGenerator
from .ingestion import CHUNKER_REGISTRY, CorpusIndex, IngestionPipeline, build_chunker
from .llm import (CompletionRequest, FakeLLM, LLM, StructuredCaller, build_llm, extract_json)
from .retrieval import ContextBuilder, Retriever, rrf, weighted_fusion
from .runtime import Runtime
from .telemetry import Tracer, configure_logging, tracer
from .types import (Chunk, ContextBlock, Document, GeneratedAnswer, PipelineResult, Query,
                    RetrievalResult, ScoredChunk, TokenUsage)

__all__ = [
    # types
    "Document", "Chunk", "Query", "ScoredChunk", "RetrievalResult", "ContextBlock",
    "GeneratedAnswer", "PipelineResult", "TokenUsage",
    # errors
    "RagArchError", "ConfigurationError", "ProviderError", "RateLimitError",
    "StructuredOutputError", "RetrievalError",
    # config / telemetry / runtime
    "CoreConfig", "Runtime", "Tracer", "tracer", "configure_logging",
    # llm
    "LLM", "CompletionRequest", "FakeLLM", "StructuredCaller", "build_llm", "extract_json",
    # ingestion / retrieval / generation
    "CorpusIndex", "IngestionPipeline", "CHUNKER_REGISTRY", "build_chunker",
    "Retriever", "ContextBuilder", "rrf", "weighted_fusion", "AnswerGenerator",
    # evaluation / dataset
    "recall_at_k", "hit_at_k", "precision_at_k", "mrr", "ndcg_at_k", "CorrectnessJudge",
    "documents", "questions", "LabeledQuestion",
]
