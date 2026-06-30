# RAG-Fusion

Multi-query generation + **Reciprocal Rank Fusion**. Generate several query reformulations, retrieve
for each, then fuse the rankings with RRF (`Σ 1/(k+rank)`) so chunks that surface across *multiple*
reformulations rise to the top.

```
query → LLM rephrasings → dense retrieve each → RRF merge → top-k → context → answer
```

The RRF step is what separates this from plain multi-query: it's a consensus vote across phrasings,
robust to one off rephrasing dragging in junk.
