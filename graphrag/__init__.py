"""GraphRAG — entity-graph retrieval with local traversal and global community map-reduce.

Implementation of Edge et al. 2024, "From Local to Global: A Graph RAG Approach to Query-Focused
Summarization" (arXiv:2404.16130), scaled honestly to the shared 14-document corpus. See README.md
for results and ARCHITECTURE.md for the data flow.

Public contract (what the benchmark imports):

    from graphrag import Config, Pipeline, KnowledgeGraph, build_graph

    graph = build_graph(runtime, runtime.corpus)      # offline, once, shareable
    pipeline = Pipeline(runtime, graph=graph)         # or omit graph for a lazy build
    retrieval, context = pipeline.retrieve(question)
    result = pipeline.answer(question)
"""
from .config import Config
from .graph import KnowledgeGraph, build_graph
from .pipeline import Pipeline

__all__ = ["Config", "Pipeline", "KnowledgeGraph", "build_graph"]
