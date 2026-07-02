"""HyDE — Hypothetical Document Embeddings (Gao et al. 2022, arXiv:2212.10496).

Query→document similarity is a harder embedding task than document→document similarity, so:
have the LLM write short hypothetical documents that would answer the question, embed those,
blend with the real query vector at `Config.query_weight`, and dense-search with the result.

Usage:

    from core import Runtime
    from hyde import Config, Pipeline

    pipeline = Pipeline(Runtime.from_env(), Config(query_weight=0.25))
    result, context = pipeline.retrieve("Who founded Veyra Systems?")
    print(result.diagnostics["hypotheses"])          # what actually steered the search
    print(pipeline.answer("Who founded Veyra Systems?").answer.text)
"""
from .config import Config
from .pipeline import Pipeline

__all__ = ["Config", "Pipeline"]
