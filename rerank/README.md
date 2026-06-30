# Rerank RAG (cross-encoder)

Two-stage retrieve-then-rerank. Dense retrieval pulls a generous candidate set (recall), then a
**cross-encoder** scores each `(query, chunk)` pair jointly and reorders for precision.

```
query → dense top-3k → cross-encoder (query+chunk together) rescoring → top-k → context → answer
```

A bi-encoder (the dense step) embeds query and chunk separately, so it can only measure coarse
similarity. A cross-encoder feeds both into one transformer and directly predicts relevance — much
sharper ranking. Most production RAG stacks add a reranker because it's the highest-ROI upgrade.
Cost: a second model pass over the candidates.
