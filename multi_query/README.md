# multi_query — query expansion + union merge

One question, phrased one way, often misses documents that say the same thing in different words.
Multi-query RAG asks an LLM for N **alternative phrasings** of the question, runs a dense search
for the original *and* every surviving variant in parallel, and merges the per-variant rankings by
round-robin interleave. The union of neighborhoods is bigger than any single query's, so lexical
mismatch between the asker's vocabulary and the corpus's vocabulary stops being fatal.

Lineage: LangChain's `MultiQueryRetriever` popularized this pattern; this package implements it
from `core` primitives only.

## How it works

1. **Expand** — `QueryExpander` prompts the LLM for `n_queries` rephrasings, parsed as a JSON
   array through `core.StructuredCaller` (validate → repair-retry). The original question is
   always kept, always first. Exact duplicates are dropped in the validator; near-duplicates are
   dropped when their embedding cosine against any already-kept query reaches `dedup_threshold`.
   If the LLM output is unusable, retrieval degrades gracefully to the original question alone.
2. **Fan out** — each kept query gets its own dense top-`per_query_k` search, run concurrently in
   a thread pool (`max_workers`) — I/O parallelism for a remote vector store/embedding service.
3. **Merge** — rankings are interleaved round-robin (rank 1 of every variant, then rank 2, ...)
   with first-seen dedup, then cut to `final_k`. Interleave, not score-sort: a global score-sort
   would let one easy phrasing monopolize the pool; interleaving guarantees every variant's head
   is represented. (Consensus-weighted fusion is the sibling architecture, `rag_fusion`.)
4. **Generate** — `core.ContextBuilder` packs the merged chunks, `core.AnswerGenerator` answers
   strictly from that context.

Everything the expansion decided — generated queries, LLM proposals, near-duplicate drops,
fallback flag, per-query hit counts — lands in `RetrievalResult.diagnostics`.

## Usage

```python
from core import Runtime
from multi_query import Config, Pipeline

pipeline = Pipeline(Runtime.from_env(), Config(n_queries=4))
result = pipeline.answer("Who founded Veyra Systems?")
print(result.answer.text)
print(result.retrieval.diagnostics["generated_queries"])
```

Offline (no network, `FakeLLM` routed on the prompt's marker phrase):

```python
from core import FakeLLM, Runtime
rt = Runtime.for_testing(llm=FakeLLM().on(
    "alternative phrasings", '["Veyra Systems founder", "who started Veyra"]'))
retrieval, context = Pipeline(rt).retrieve("Who founded Veyra Systems?")
```

## Files

| File | Role |
|---|---|
| `config.py` | Frozen `Config` — every tunable (fan-out width/depth, dedup, workers, context) |
| `prompts.py` | The expansion prompt (marker phrase: *"alternative phrasings"*) |
| `expander.py` | `QueryExpander` — StructuredCaller + validator + semantic dedup |
| `retriever.py` | Parallel fan-out + round-robin interleave merge |
| `pipeline.py` | `Pipeline` — composition, diagnostics, tracing |

## Honest limitation: 0% on multi-hop

No rephrasing of a question retrieves a **bridge document** the question never mentions. "Which
city is the founder of Veyra Systems from?" can be rephrased forever without ever containing the
founder's *name* — the term the bridge document matches on. In this repo's benchmark the whole
query-transform family (multi_query, rag_fusion, hyde) scored **0% on multi-hop questions** for
exactly this reason. Expansion widens the neighborhood of the question; it cannot walk to a
second hop. Structural/iterative architectures (agentic, RAPTOR, graphrag) are the fix.

Also budget for cost: each question costs 1 LLM call plus up to `n_queries + 1` dense searches.
