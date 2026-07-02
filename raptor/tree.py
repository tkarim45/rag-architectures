"""The RAPTOR tree — the offline artifact that online retrieval reads over — and its builder.

A tree is a flat map of nodes indexed by id. Level 0 nodes are leaves (one per source document);
each higher level holds LLM summaries of soft clusters of the level below. Two properties are
load-bearing for retrieval:

* every node carries its own **embedding**, because collapsed-tree retrieval scores all nodes of
  all levels in a single similarity pass; and
* every node carries **source_doc_ids** — the union of the leaf documents underneath it. This is
  the provenance that lets a hit on a level-2 summary credit *all* the documents it abstracts,
  which is exactly how RAPTOR wins on multi-hop questions whose evidence is spread across docs.

`build_tree` is the module-level offline entrypoint the benchmark calls once and shares across
pipelines (per the core package contract).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from core import Document, Runtime

from .clustering import cluster_level
from .config import Config
from .summarizer import summarize_cluster


@dataclass(frozen=True, eq=False)
class RaptorNode:
    """One node of the tree.

    `eq=False` because the embedding is a NumPy array (ambiguous `==`); node identity is
    `node_id`, and the tree only ever looks nodes up by id anyway.
    """

    node_id: str
    level: int                                 # 0 = leaf; higher = more abstract summary
    text: str                                  # document text (leaf) or LLM summary (internal)
    embedding: np.ndarray                      # (dim,) float32, L2-normalized by the embedder
    children_ids: tuple[str, ...] = ()         # empty for leaves
    source_doc_ids: tuple[str, ...] = ()       # union of leaf provenance, order-preserving

    @property
    def is_leaf(self) -> bool:
        return self.level == 0

    @property
    def approx_tokens(self) -> int:
        """chars/4 token estimate — the same approximation the retrieval budget uses."""
        return max(1, len(self.text) // 4)


@dataclass(frozen=True)
class RaptorTree:
    """Immutable node store plus the level index retrieval and diagnostics read."""

    nodes: dict[str, RaptorNode] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.nodes)

    def node(self, node_id: str) -> RaptorNode:
        return self.nodes[node_id]

    def all_nodes(self) -> list[RaptorNode]:
        """Every node of every level — the collapsed-tree retrieval candidate set."""
        return list(self.nodes.values())

    def leaves(self) -> list[RaptorNode]:
        return self.nodes_at(0)

    def nodes_at(self, level: int) -> list[RaptorNode]:
        return [n for n in self.nodes.values() if n.level == level]

    @property
    def max_level(self) -> int:
        return max((n.level for n in self.nodes.values()), default=0)

    @property
    def num_levels(self) -> int:
        return self.max_level + 1 if self.nodes else 0

    def embedding_matrix(self) -> tuple[list[RaptorNode], np.ndarray]:
        """(nodes, (n, dim) matrix) in matching order — one dot product scores the whole tree,
        since every embedding is L2-normalized at the embedder boundary."""
        ordered = self.all_nodes()
        if not ordered:
            return [], np.zeros((0, 0), dtype=np.float32)
        return ordered, np.stack([n.embedding for n in ordered]).astype(np.float32)

    def describe(self) -> dict[str, int]:
        """Shape summary for logs, traces, and benchmark manifests."""
        return {f"level_{lvl}": len(self.nodes_at(lvl)) for lvl in range(self.num_levels)}


# ------------------------------------------------------------------------------------------
# Offline construction
# ------------------------------------------------------------------------------------------

def build_tree(runtime: Runtime, documents: Sequence[Document],
               config: Config | None = None) -> RaptorTree:
    """Build the RAPTOR tree bottom-up: embed leaves, then cluster → summarize → embed upward.

    Leaves are one node per *document* (not per chunk): this corpus's documents are short
    passages, so document granularity matches the paper's "100-token chunk" leaves without an
    extra chunking layer. Each level softly clusters the level below (GMM + BIC, see
    `clustering.py`), summarizes each cluster with the LLM (see `summarizer.py`), embeds the
    summaries, and recurses until a single root remains, `config.max_levels` is reached, or a
    level stops shrinking (in which case the remainder is folded into one root cluster so
    construction always terminates with a rooted tree).
    """
    config = config or Config()
    docs = list(documents)
    with runtime.tracer.span("raptor.build_tree", documents=len(docs),
                             max_levels=config.max_levels) as build_span:
        nodes: dict[str, RaptorNode] = {}
        if not docs:
            return RaptorTree(nodes={})

        with runtime.tracer.span("raptor.embed_leaves", texts=len(docs)):
            leaf_matrix = runtime.embedder.embed_texts([d.text for d in docs])
        current: list[RaptorNode] = []
        for i, doc in enumerate(docs):
            leaf = RaptorNode(node_id=f"L0.{doc.doc_id}", level=0, text=doc.text,
                              embedding=leaf_matrix[i], children_ids=(),
                              source_doc_ids=(doc.doc_id,))
            nodes[leaf.node_id] = leaf
            current.append(leaf)

        for level in range(1, config.max_levels + 1):
            if len(current) <= 1:
                break
            with runtime.tracer.span("raptor.cluster_level", level=level,
                                     nodes=len(current)) as cluster_span:
                embeddings = np.stack([n.embedding for n in current]).astype(np.float32)
                clusters = cluster_level(embeddings, config)
                if len(clusters) >= len(current):
                    # No reduction (e.g. BIC insists on many tiny components): force one root
                    # cluster rather than recursing forever on a level that will not shrink.
                    clusters = [tuple(range(len(current)))]
                cluster_span.set("clusters", len(clusters))
                cluster_span.set("cluster_sizes", [len(c) for c in clusters])

            summaries: list[str] = []
            memberships: list[tuple[int, ...]] = []
            for member_indices in clusters:
                texts = [current[i].text for i in member_indices]
                summaries.append(summarize_cluster(runtime.llm, texts, config, runtime.tracer))
                memberships.append(member_indices)

            with runtime.tracer.span("raptor.embed_summaries", texts=len(summaries)):
                summary_matrix = runtime.embedder.embed_texts(summaries)

            next_level: list[RaptorNode] = []
            for j, (summary, member_indices) in enumerate(zip(summaries, memberships)):
                node = RaptorNode(
                    node_id=f"L{level}.C{j}", level=level, text=summary,
                    embedding=summary_matrix[j],
                    children_ids=tuple(current[i].node_id for i in member_indices),
                    source_doc_ids=_union_source_docs(current, member_indices))
                nodes[node.node_id] = node
                next_level.append(node)
            current = next_level

        tree = RaptorTree(nodes=nodes)
        build_span.set("shape", tree.describe())
        build_span.set("total_nodes", len(tree))
        return tree


def _union_source_docs(level_nodes: list[RaptorNode],
                       member_indices: tuple[int, ...]) -> tuple[str, ...]:
    """Order-preserving union of the members' leaf provenance — the multi-doc credit a summary
    node carries into retrieval."""
    seen: set[str] = set()
    out: list[str] = []
    for i in member_indices:
        for doc_id in level_nodes[i].source_doc_ids:
            if doc_id not in seen:
                seen.add(doc_id)
                out.append(doc_id)
    return tuple(out)
