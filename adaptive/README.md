# Adaptive RAG (router)

No single pipeline is best for every query, so **classify first, then dispatch**. A lightweight LLM
router labels the query and sends it to the cheapest sufficient retriever.

```
query → LLM router ─┬─ simple    → dense
                    ├─ multi_hop → GraphRAG
                    └─ broad     → RAG-Fusion        → context → answer
```

This is the emerging enterprise default: a cheap classifier in front routes the ~80% easy queries
to fast dense search and reserves expensive graph/fusion paths for the queries that need them. The
risk is router error — a misrouted multi-hop question lands on dense and fails.
