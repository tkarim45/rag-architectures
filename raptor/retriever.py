"""RAPTOR retriever — the online half. Embed the query, score every tree node (leaves and parent
summaries together) by cosine, then walk the nodes in descending score and expand each one's
`covers` into a deduped document list until we have k docs. Because a parent summary covers a whole
cluster, one strong match on a summary can surface several related documents at once."""
from __future__ import annotations

import numpy as np

from common import providers


def retrieve(query: str, raptor, k: int = 5) -> tuple[list[str], str]:
    qv = providers.embed([query])[0]
    mat = np.vstack([n.vec for n in raptor.nodes])
    scores = mat @ qv
    order = np.argsort(scores)[::-1]

    docs: list[str] = []
    for idx in order:
        for did in raptor.nodes[idx].covers:
            if did not in docs:
                docs.append(did)
        if len(docs) >= k:
            break
    docs = docs[:k]

    ctx = "\n\n".join(raptor.doc_text[d] for d in docs if d in raptor.doc_text)
    return docs, ctx
