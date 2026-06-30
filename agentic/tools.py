"""The two retrieval tools the agent drives. Each wraps another architecture's retriever behind one
uniform shape — (doc_ids, context_block) — so the agent loop can call either without caring how the
underlying retrieval works.

  * vector_search — dense semantic search over the chunk index (chunk-level, collapsed to docs).
  * graph_search  — entity-graph traversal for multi-hop questions (already doc-level upstream).
"""
from __future__ import annotations

from naive import retrieve as dense_retrieve
from graphrag import retrieve as graph_retrieve
from common.retrieval import context, to_docs


def vector_search(query: str, index, k: int = 4) -> tuple[list[str], str]:
    """Semantic search: top-k chunks → (unique doc ids, context block)."""
    cids = dense_retrieve(query, index, k)
    return to_docs(cids, index), context(cids, index)


def graph_search(query: str, graph, k: int = 4) -> tuple[list[str], str]:
    """Entity-graph traversal: returns (doc ids, context block) already at doc level."""
    return graph_retrieve(query, graph, k)
