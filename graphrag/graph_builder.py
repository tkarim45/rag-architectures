"""Graph construction — the OFFLINE half of GraphRAG, run once at build time.

For every document we extract its entities (one LLM call each), invert that into an entity->docs
index, then add one node per doc and an edge between every pair of docs that share at least one
entity. The result is a doc-doc graph whose edges encode "these two documents talk about the same
thing" — the structure the online query path traverses instead of ranking by vector similarity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import networkx as nx

from .extractor import extract_entities


@dataclass
class Graph:
    doc_text: dict          # doc_id -> full document text (what the generator ultimately reads)
    doc_entities: dict      # doc_id -> set[str] of entities mentioned in that doc
    entity_docs: dict       # entity -> set[doc_id] (the inverted index used to seed a query)
    g: nx.Graph             # nodes = doc_ids, edges = shared at least one entity


def build(docs) -> Graph:
    """Build the doc-doc graph from a corpus. OFFLINE: one entity-extraction LLM call per doc.

    docs: iterable of objects with `.id` and `.text` (e.g. common.corpus.Doc).
    """
    doc_text: dict = {}
    doc_entities: dict = {}
    entity_docs: dict = {}

    for d in docs:
        ents = extract_entities(d.text)
        doc_text[d.id] = d.text
        doc_entities[d.id] = ents
        for e in ents:
            entity_docs.setdefault(e, set()).add(d.id)

    g = nx.Graph()
    g.add_nodes_from(doc_text)                       # one node per document
    for doc_ids in entity_docs.values():             # for each entity, link every pair sharing it
        for a, b in combinations(sorted(doc_ids), 2):
            g.add_edge(a, b)

    return Graph(doc_text=doc_text, doc_entities=doc_entities, entity_docs=entity_docs, g=g)
