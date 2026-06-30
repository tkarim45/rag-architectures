# GraphRAG

Structure the knowledge itself. Extract entities from every document (with an LLM), connect docs
that share entities into a graph, and answer by **traversing** that graph rather than ranking by
similarity.

```
build:  docs ─→ entity extraction ─→ entity→doc index + doc–doc graph (shared-entity edges)
query:  query entities → seed docs → BFS expand N hops → ranked docs → context → answer
```

**Why it beats vector search on multi-hop:** "Who founded the company behind the database Quorrel
uses?" never lexically/semantically resembles the *founder* document, so flat retrieval misses it.
The graph walks Quorrel → Talix → Brightfen → founder by following shared-entity edges. **Cost:**
an LLM entity-extraction pass per document at build time.
