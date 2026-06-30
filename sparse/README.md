# Sparse RAG (BM25)

Classic lexical retrieval — rank chunks by BM25 keyword-overlap score, no embeddings at all.

```
query → tokenize → BM25 over chunk tokens → top-k → context → answer
```

**Strength:** exact-term matches (names, codes, rare words) that dense embeddings can blur.
**Weakness:** no semantics — a paraphrase with different words scores zero. Included as the
counterpart to dense, and as one half of `hybrid`.
