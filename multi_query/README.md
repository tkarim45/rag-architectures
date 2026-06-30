# Multi-Query RAG

Ask the LLM to rephrase the question several ways, retrieve for every variant, and union the hits.
Broadens lexical/semantic coverage when a single phrasing under-retrieves.

```
query → LLM rephrasings {q, q1, q2, q3} → dense retrieve each → union (best rank) → top-k → answer
```

**Cost:** one extra LLM call for the rephrasings + N dense searches. **Vs RAG-Fusion:** same idea,
but multi-query unions by best rank while RAG-Fusion fuses with RRF.
