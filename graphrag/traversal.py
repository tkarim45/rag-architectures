"""Graph traversal — the ONLINE expansion step. Given the seed documents a query lands on, walk the
shared-entity graph outward by breadth-first search and collect documents in hop order: seeds first,
then everything one hop away, then two hops, and so on. This is what lets GraphRAG answer multi-hop
questions — it chains Quorrel -> Talix -> Brightfen -> founder by following edges, not similarity.
"""
from __future__ import annotations

from collections import deque

from .graph_builder import Graph


def traverse(seeds: set[str], graph: Graph, k: int, hops: int) -> list[str]:
    """BFS from `seeds` over `graph.g`, returning up to `k` doc-ids in hop order (seeds first).

    Visits seeds at distance 0, their neighbors at distance 1, ... up to `hops` away, deduping and
    preserving discovery order. Seeds that aren't graph nodes are still emitted (they're valid docs).
    """
    ordered: list[str] = []
    seen: set[str] = set()
    queue: deque[tuple[str, int]] = deque()

    for s in sorted(seeds):               # seeds occupy hop 0, in stable (sorted) order
        if s not in seen:
            seen.add(s)
            ordered.append(s)
            queue.append((s, 0))

    while queue:
        node, dist = queue.popleft()
        if dist >= hops or node not in graph.g:
            continue
        for nbr in graph.g.neighbors(node):
            if nbr not in seen:
                seen.add(nbr)
                ordered.append(nbr)
                queue.append((nbr, dist + 1))

    return ordered[:k]
