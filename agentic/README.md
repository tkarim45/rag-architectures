# Agentic RAG

Retrieval as an agent loop. The LLM holds retrieval *tools* (vector search, graph traversal) and
drives its own investigation: pick a tool and query, read the result, decide whether to search again
or stop, then answer from everything it gathered.

```
loop (≤ N steps):  LLM → {vector_search | graph_search | finish} → run tool → append to notes
finish:            answer from accumulated notes
```

The most flexible pattern — it can chain its own multi-hop lookups and adapt mid-query — and the
most expensive: several LLM round-trips per question, with a real risk of loops, dead ends, or
stopping too early. This repo measures whether that flexibility actually pays off versus the cheaper
fixed pipelines.
