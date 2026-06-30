"""GraphRAG retriever — the ONLINE query path. Doc-level: it returns whole documents, not chunks.

Flow: extract the entities mentioned in the query, look them up in the pre-built entity->doc index
to find the *seed* documents, BFS-expand those seeds across shared-entity edges, then stitch the
resulting documents into a numbered context string for the generator. The graph it reads was built
offline once (graph_builder.build); nothing here re-extracts the corpus.

`retrieve(query, graph, k=5, hops=2) -> (ranked_doc_ids, context_string)` — this exact signature is
imported by the adaptive/ and agentic/ architectures, so keep it stable.
"""
from __future__ import annotations

from .extractor import extract_entities
from .graph_builder import Graph
from .traversal import traverse


def retrieve(query: str, graph: Graph, k: int = 5, hops: int = 2) -> tuple[list[str], str]:
    q_entities = extract_entities(query)

    seeds: set[str] = set()
    for e in q_entities:
        seeds |= graph.entity_docs.get(e, set())

    if not seeds:                                  # fallback: substring-match query entities to known
        for qe in q_entities:                      # entities (handles paraphrase / partial mentions)
            for known, docs in graph.entity_docs.items():
                if qe and (qe in known or known in qe):
                    seeds |= docs

    ranked = traverse(seeds, graph, k, hops)
    context = "\n\n".join(f"[{i + 1}] {graph.doc_text[d]}" for i, d in enumerate(ranked))
    return ranked, context
