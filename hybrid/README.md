# Hybrid RAG (dense + sparse + RRF)

Run dense (semantic) and sparse (BM25 lexical) retrieval independently, then merge their rankings
with **Reciprocal Rank Fusion**: `score(doc) = Σ 1/(k + rank)` across both lists.

```
query ─┬─ dense  top-2k ─┐
       └─ BM25   top-2k ─┴─ RRF merge → top-k → context → answer
```

**Why it works:** the two retrievers fail on different queries (dense on exact rare terms, sparse
on paraphrase), so fusing them covers both. One of the best effort-to-quality ratios in RAG.
