"""Community detection and summarization — the paper's "global" half of the offline build.

Microsoft GraphRAG's key move beyond plain entity graphs (Edge et al. 2024) is hierarchical
community detection over the entity graph plus an LLM-written report per community: corpus-level
questions are then answered from community summaries instead of individual passages. At this
corpus scale (14 documents, a few dozen entities) one flat Louvain partition is the honest
equivalent of the paper's Leiden hierarchy — there is not enough graph for multiple levels to
differ, so we do not pretend otherwise.

Louvain runs on the *undirected weighted projection* of the MultiDiGraph: community structure is
about how densely entities co-occur in relations, not about edge direction, and parallel edges
collapse into an integer weight so repeatedly co-mentioned pairs bind their communities tighter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx
from networkx.algorithms.community import louvain_communities

from core import LLM, Tracer

from .config import Config
from .prompts import COMMUNITY_SUMMARY_PROMPT


@dataclass(frozen=True)
class Community:
    """One detected entity cluster plus its LLM-written report and document provenance."""

    community_id: int
    entities: tuple[str, ...]      # normalized entity names (graph node keys)
    doc_ids: tuple[str, ...]       # union of member entities' provenance docs
    summary: str                   # LLM report; empty for communities below min size


def undirected_projection(graph: nx.MultiDiGraph) -> nx.Graph:
    """Collapse the typed multi-digraph into an undirected weighted graph for Louvain. Weight =
    number of parallel relation edges between the pair (co-mention strength)."""
    projection = nx.Graph()
    projection.add_nodes_from(graph.nodes)
    for source, target in graph.edges():
        if projection.has_edge(source, target):
            projection[source][target]["weight"] += 1
        else:
            projection.add_edge(source, target, weight=1)
    return projection


def detect_communities(graph: nx.MultiDiGraph, config: Config) -> list[tuple[str, ...]]:
    """Louvain partition of the entity graph, deterministic via the configured seed. Returns
    member-name tuples sorted for reproducible community ids across builds."""
    if graph.number_of_nodes() == 0:
        return []
    partition = louvain_communities(undirected_projection(graph), weight="weight",
                                    seed=config.louvain_seed)
    members = [tuple(sorted(community)) for community in partition]
    members.sort()                               # stable community_id assignment
    return members


def _format_entities(graph: nx.MultiDiGraph, names: Iterable[str]) -> str:
    lines = []
    for name in names:
        node = graph.nodes[name]
        description = "; ".join(node.get("descriptions", ())) or "(no description)"
        lines.append(f"- {node.get('display_name', name)} ({node.get('type', 'other')}): "
                     f"{description}")
    return "\n".join(lines)


def _format_relations(graph: nx.MultiDiGraph, names: set[str]) -> str:
    lines = []
    for source, target, data in graph.edges(data=True):
        if source in names and target in names:
            src = graph.nodes[source].get("display_name", source)
            tgt = graph.nodes[target].get("display_name", target)
            line = f"- {src} -[{data.get('type', 'related_to')}]-> {tgt}"
            if data.get("description"):
                line += f": {data['description']}"
            lines.append(line)
    return "\n".join(lines) or "(no internal relations)"


def build_communities(graph: nx.MultiDiGraph, llm: LLM, config: Config,
                      tracer: Tracer) -> tuple[Community, ...]:
    """Detect communities and write one LLM summary per community of at least
    `config.min_community_size` members (singletons carry no relational structure — skipping the
    call is the honest scale-down, but the community is still recorded for provenance)."""
    communities: list[Community] = []
    with tracer.span("graphrag.communities", nodes=graph.number_of_nodes()) as span:
        for community_id, members in enumerate(detect_communities(graph, config)):
            doc_ids = sorted({doc for name in members
                              for doc in graph.nodes[name].get("doc_ids", ())})
            summary = ""
            if len(members) >= config.min_community_size:
                with tracer.span("graphrag.summarize_community", community_id=community_id,
                                 entities=len(members)):
                    summary = llm.complete_text(
                        COMMUNITY_SUMMARY_PROMPT.format(
                            entities=_format_entities(graph, members),
                            relations=_format_relations(graph, set(members))),
                        max_tokens=config.summary_max_tokens).strip()
            communities.append(Community(community_id=community_id, entities=members,
                                         doc_ids=tuple(doc_ids), summary=summary))
        span.set("communities", len(communities))
        span.set("summarized", sum(1 for c in communities if c.summary))
    return tuple(communities)
