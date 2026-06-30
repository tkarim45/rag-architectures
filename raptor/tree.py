"""RAPTOR tree — the offline structure retrieval reads over.

Two layers, built once:
  * leaves   — one Node per source document (covers its own id).
  * parents  — one Node per cluster of >=2 leaves: an LLM summary that *covers* every member doc.

Retrieval scores leaves and parents together; a hit on a parent summary expands into all the
documents it covers, which is what lets a single match pull in a whole related cluster.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from common import providers

from .clustering import cluster
from .summarizer import summarize


@dataclass
class Node:
    text: str               # the document text (leaf) or the cluster summary (parent)
    covers: list[str]       # doc ids this node expands into when retrieved
    vec: object             # L2-normalized embedding of `text`


@dataclass
class Raptor:
    nodes: list[Node]
    doc_text: dict          # {doc_id: text} — used to build context from a node's covers


def build(docs, n_clusters: int = 4) -> Raptor:
    docs = list(docs)
    texts = [d.text for d in docs]
    vecs = providers.embed(texts)

    # leaves: one node per source document
    nodes = [Node(text=d.text, covers=[d.id], vec=vecs[i]) for i, d in enumerate(docs)]

    # parents: summarize each cluster of >=2 leaves into a higher-level node
    labels = cluster(vecs, n_clusters)
    for c in set(labels):
        members = [i for i, lbl in enumerate(labels) if lbl == c]
        if len(members) < 2:
            continue
        summary = summarize([docs[i].text for i in members])
        nodes.append(Node(
            text=summary,
            covers=[docs[i].id for i in members],
            vec=providers.embed([summary])[0],
        ))

    return Raptor(nodes=nodes, doc_text={d.id: d.text for d in docs})
