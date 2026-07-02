"""Corrective RAG (CRAG) — grade the retrieval, then act on the grade.

Yan et al. 2024, "Corrective Retrieval Augmented Generation" (arXiv:2401.15884): a retrieval
evaluator grades each retrieved passage, the grades aggregate to a per-query verdict
(CORRECT / INCORRECT / AMBIGUOUS), and the verdict picks the action — refine the retrieved
knowledge, discard it and search wider on a rewritten query, or combine both.

Public surface (the standard package contract):

    from corrective import Config, Pipeline

    pipeline = Pipeline(runtime, Config(), index=shared_index)
    result, context = pipeline.retrieve("Who founded Veyra Systems?")
    pipeline_result = pipeline.answer("Who founded Veyra Systems?")
"""
from .config import Config
from .pipeline import Pipeline

__all__ = ["Config", "Pipeline"]
