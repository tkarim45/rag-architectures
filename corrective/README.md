# Corrective RAG (CRAG)

Don't trust the first retrieval — grade it. Retrieve (hybrid), have the LLM judge each chunk's
relevance, and if too few clear the bar, **rewrite the query and retrieve again**.

```
hybrid retrieve → LLM grades each chunk ─ enough relevant? ─┬─ yes → answer
                                                            └─ no  → rewrite query → RAG-Fusion re-retrieve → answer
```

The original CRAG falls back to **web search** when retrieval confidence is low; here (closed
corpus) the corrective action is a query-rewrite + fusion re-retrieval. The point is the same: a
self-check + recovery step instead of blindly answering from a bad first hit.
