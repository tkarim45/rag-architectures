# Naive RAG

The baseline. Embed the query, retrieve the top-k chunks by cosine similarity, stuff them into the
prompt, generate. No query transformation, no reranking, no validation, no fallback.

```
query → embed → top-k dense retrieval → context → LLM answer
```

**Strength:** fast, cheap, simple. **Weakness:** one shot — if dense retrieval misses (vocabulary
mismatch, multi-hop question), there's no recovery. Everything else in this repo is an attempt to
fix a specific failure of this baseline.
