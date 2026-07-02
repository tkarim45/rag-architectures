# GraphRAG

Entity-graph retrieval with two search modes, after Microsoft's GraphRAG — Edge et al. 2024,
*"From Local to Global: A Graph RAG Approach to Query-Focused Summarization"*
(arXiv:2404.16130) — scaled honestly to the shared 14-document corpus.

Instead of ranking chunks by vector similarity, GraphRAG builds a **typed entity graph** offline
(LLM-extracted entities and relations, merged across documents, with per-edge document
provenance), detects **Louvain communities** over it, and writes an **LLM summary per community**.
Online, questions route to one of two searches from the paper:

- **Local search** — entity-centric questions. Link the question's entities to graph nodes
  (LLM extraction first, lexical name-match fallback merged in), expand the neighborhood up to
  `max_hops`, and rank the provenance documents of every entity and relation reached, decayed by
  hop distance.
- **Global search** — corpus-level questions. Map: rate each community summary's relevance to the
  question (structured 0–10 call). Reduce: keep the top communities and rank their provenance
  documents by summed rating.

`mode="auto"` adds a one-call router that picks local vs global per question.

## Usage

```python
from core import Runtime
from graphrag import Config, Pipeline, build_graph

runtime = Runtime.from_env()

# offline artifact — build once, share across pipelines (the benchmark does exactly this)
graph = build_graph(runtime, runtime.corpus)

pipeline = Pipeline(runtime, Config(mode="auto"), graph=graph)
retrieval, context = pipeline.retrieve("Who founded the company that makes Talix?")
result = pipeline.answer("Who founded the company that makes Talix?")
print(result.answer.text)
print(retrieval.diagnostics["traversal_paths"])   # e.g. "Talix -[created_by]-> Brightfen"
```

Omit `graph=` / `index=` and the pipeline builds both lazily from `runtime.corpus` (the doc-id →
chunk index uses the `"whole"` chunker — GraphRAG never vector-searches it; the graph decides
*which* documents, the index just hands them to the generator).

Runs fully offline under `Runtime.for_testing()` with a `FakeLLM` routed on the prompt markers in
`prompts.py`.

## Files

| File | Role |
|---|---|
| `config.py` | Frozen `Config` — mode, hops, top-k, community and context budgets |
| `prompts.py` | Every LLM touchpoint (extraction, summary, entity linking, rating, router) |
| `extractor.py` | Offline per-doc entity/relation extraction via `StructuredCaller` + validator |
| `graph.py` | `KnowledgeGraph` (MultiDiGraph + entity→docs index) and `build_graph` |
| `communities.py` | Louvain detection on the undirected projection + community summaries |
| `local_search.py` | Entity linking → hop-bounded traversal → provenance scoring |
| `global_search.py` | Community map (rate) / reduce (keep top, rank docs) |
| `retriever.py` | Router + orchestration; doc ids → `ScoredChunk`s; rich diagnostics |
| `pipeline.py` | Framework contract: `retrieve()` / `answer()`, lazy artifacts |

## The honest finding

At this corpus scale GraphRAG is **gated by extraction quality and hop limits**: it scored
**67% overall / 25% multi-hop** in the last benchmark run. Traversal genuinely *does* reach bridge
documents — the diagnostics show 2-hop paths like `Quorrel -[stores state in]-> Talix -[created
by]-> Brightfen` resolving to the exact bridge doc — but 3-hop chains can slip past the default
`max_hops=2`, and any entity or relation the extractor misses simply does not exist in the graph,
silently amputating every path through it. That is the real-world lesson this package is built to
teach: **graph quality = extraction quality**. A knowledge graph is not a retrieval upgrade you
bolt on; it is only as good as the noisy LLM stage that built it, and its failure mode is silence,
not a low score.

Raising `max_hops` recovers longer chains at the cost of precision (each hop lets more of the
corpus vote); the tuning table in `ARCHITECTURE.md` walks the trade-offs.
