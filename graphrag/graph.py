"""The knowledge graph — GraphRAG's offline artifact.

Unlike the doc-doc "shared entity" shortcut some GraphRAG reimplementations take, this follows the
paper's data model: nodes are *entities* (typed, with descriptions merged across every document
that mentions them) and edges are *typed relations* carrying per-document provenance. Documents are
reattached at query time through the entity→docs inverted index, which is what lets a traversal
path ("Quorrel -[stores state in]-> Talix -[created by]-> Brightfen") resolve back to the exact
bridge documents a multi-hop question needs.

Built once by `build_graph`, shared read-only by every Pipeline instance — the same offline/online
split as `CorpusIndex`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import networkx as nx

from core import Document, Runtime

from .communities import Community, build_communities
from .config import Config
from .extractor import DocumentExtraction, EntityRelationExtractor


@dataclass
class KnowledgeGraph:
    """Entity graph + inverted index + community reports. Node keys are normalized entity names
    (see `extractor.normalize_entity_name`); node attrs: `display_name`, `type`, `descriptions`,
    `doc_ids`; edge attrs: `type`, `description`, `doc_id` (provenance of the stating document)."""

    graph: nx.MultiDiGraph
    entity_docs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    communities: tuple[Community, ...] = ()

    # ---- lookups -----------------------------------------------------------------------

    @property
    def entity_names(self) -> list[str]:
        return list(self.graph.nodes)

    def has_entity(self, name: str) -> bool:
        return self.graph.has_node(name)

    def display_name(self, name: str) -> str:
        return str(self.graph.nodes[name].get("display_name", name))

    def docs_of(self, name: str) -> tuple[str, ...]:
        """Provenance documents for an entity (which docs mention it)."""
        return self.entity_docs.get(name, ())

    def neighbors(self, name: str) -> set[str]:
        """Undirected neighborhood — traversal ignores edge direction ("founded by" must be
        walkable both ways) but keeps direction in the reported paths."""
        return set(self.graph.successors(name)) | set(self.graph.predecessors(name))

    def stats(self) -> dict[str, Any]:
        """Compact shape summary for diagnostics and trace spans."""
        return {"entities": self.graph.number_of_nodes(),
                "relations": self.graph.number_of_edges(),
                "communities": len(self.communities),
                "summarized_communities": sum(1 for c in self.communities if c.summary)}


def _merge_extractions(extractions: Sequence[DocumentExtraction]) -> nx.MultiDiGraph:
    """Fold per-document extractions into one graph. Duplicate entities across documents merge on
    the normalized name; their type is upgraded from "other" when a later doc is more specific and
    their descriptions accumulate (deduplicated) — the paper's element-summarization step, scaled
    down to concatenation since descriptions here are single sentences."""
    graph = nx.MultiDiGraph()
    for extraction in extractions:
        for entity in extraction.entities:
            if not graph.has_node(entity.name):
                graph.add_node(entity.name, display_name=entity.display_name, type=entity.type,
                               descriptions=[], doc_ids=set())
            node = graph.nodes[entity.name]
            if node["type"] == "other" and entity.type != "other":
                node["type"] = entity.type
            if entity.description and entity.description not in node["descriptions"]:
                node["descriptions"].append(entity.description)
            node["doc_ids"].add(extraction.doc_id)
        for relation in extraction.relations:
            graph.add_edge(relation.source, relation.target, type=relation.type,
                           description=relation.description, doc_id=extraction.doc_id)
    # Freeze accumulators into deterministic tuples once merging is done.
    for _, node in graph.nodes(data=True):
        node["descriptions"] = tuple(node["descriptions"])
        node["doc_ids"] = tuple(sorted(node["doc_ids"]))
    return graph


def build_graph(runtime: Runtime, documents: Sequence[Document],
                config: Config | None = None) -> KnowledgeGraph:
    """OFFLINE builder: one extraction call per document, merge, detect communities, summarize.

    This is the expensive artifact (N extraction calls + one summary per community); the benchmark
    calls it once and injects the result into every Pipeline so all runs share identical graphs.
    """
    config = config or Config()
    with runtime.tracer.span("graphrag.build_graph", docs=len(documents)) as span:
        extractor = EntityRelationExtractor(llm=runtime.llm, config=config,
                                            tracer=runtime.tracer)
        extractions = [extractor.extract(document) for document in documents]
        graph = _merge_extractions(extractions)
        entity_docs = {name: data["doc_ids"] for name, data in graph.nodes(data=True)}
        communities = build_communities(graph, runtime.llm, config, runtime.tracer)
        kg = KnowledgeGraph(graph=graph, entity_docs=entity_docs, communities=communities)
        for key, value in kg.stats().items():
            span.set(key, value)
    return kg
