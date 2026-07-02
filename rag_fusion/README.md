# rag_fusion — query expansion + Reciprocal Rank Fusion

RAG-Fusion (Rackauckas 2024) asks an LLM to **broaden and diversify** the question into several
search queries, runs a dense search per query in parallel, and combines the per-query rankings
with Reciprocal Rank Fusion (RRF, Cormack et al. 2009): `score(d) = Σ 1/(rrf_k + rank)`. The
result is a *consensus* ranking — a chunk that shows up across many query angles beats a chunk
that tops exactly one.

## How it works

1. **Broaden** — `QueryExpander` prompts the LLM for `n_queries` diversified queries (sub-aspects,
   related entities, alternate vocabulary), parsed as a JSON array through `core.StructuredCaller`
   (validate → repair-retry). The original question is always kept, always first. Exact duplicates
   die in the validator; near-duplicates die when their embedding cosine against any kept query
   reaches `dedup_threshold` — under RRF a duplicate query is actively harmful, it double-counts
   the same votes. Unusable LLM output degrades gracefully to the original question alone.
2. **Fan out** — each kept query gets its own dense top-`per_query_k` search, run concurrently in
   a thread pool (`max_workers`) — I/O parallelism for a remote vector store/embedding service.
3. **Fuse** — `core.retrieval.fusion.rrf` with configurable `rrf_k`, then cut to `final_k`.
   Rank-based, so scores across queries never need to be on comparable scales.
4. **Generate** — `core.ContextBuilder` packs the fused chunks, `core.AnswerGenerator` answers
   strictly from that context.

Everything the expansion and fusion decided — generated queries, drops, fallback flag, per-query
hit counts, fused pool size, `rrf_k` — lands in `RetrievalResult.diagnostics`.

## vs. `multi_query`

Same fan-out skeleton, opposite merge philosophy:

| | `multi_query` | `rag_fusion` |
|---|---|---|
| Merge | Round-robin interleave | Reciprocal Rank Fusion |
| Bet | *Diversity*: every variant's head deserves a slot | *Consensus*: documents surfacing across MANY variants win |
| A doc found by all N queries | One slot, position from its first variant | Aggregated votes push it to the top |
| A doc found by one query only | Guaranteed early slot | Damped by `rrf_k`, may fall out |
| Risk | One drifted variant injects noise into the pool | Genuinely relevant one-query docs get outvoted |

## Usage

```python
from core import Runtime
from rag_fusion import Config, Pipeline

pipeline = Pipeline(Runtime.from_env(), Config(rrf_k=60))
result = pipeline.answer("Who founded Veyra Systems?")
print(result.answer.text)
print(result.retrieval.diagnostics["generated_queries"])
```

Offline (no network, `FakeLLM` routed on the prompt's marker phrase):

```python
from core import FakeLLM, Runtime
rt = Runtime.for_testing(llm=FakeLLM().on(
    "broaden and diversify", '["Veyra Systems founder", "who started Veyra"]'))
retrieval, context = Pipeline(rt).retrieve("Who founded Veyra Systems?")
```

## Files

| File | Role |
|---|---|
| `config.py` | Frozen `Config` — every tunable (fan-out, `rrf_k`, dedup, workers, context) |
| `prompts.py` | The broadening prompt (marker phrase: *"broaden and diversify"*) |
| `expander.py` | `QueryExpander` — StructuredCaller + validator + semantic dedup |
| `retriever.py` | Parallel fan-out + RRF fusion (via `core.retrieval.fusion.rrf`) |
| `pipeline.py` | `Pipeline` — composition, diagnostics, tracing |

## Honest limitation: 0% on multi-hop

Shared with the whole query-transform family: no rephrasing or broadening of a question retrieves
a **bridge document** the question never names. "Which city is the founder of Veyra Systems
from?" broadened five ways still never contains the founder's *name* — the term the bridge
document matches on. In this repo's benchmark this family (multi_query, rag_fusion, hyde) scored
**0% on multi-hop questions** for exactly this reason. Consensus scoring can even make it worse:
the variants all agree on the same first-hop documents. Structural/iterative architectures
(agentic, RAPTOR, graphrag) are the fix.

Also budget for cost: each question costs 1 LLM call plus up to `n_queries + 1` dense searches.

## References

- Rackauckas, Z. (2024). *RAG-Fusion: a New Take on Retrieval-Augmented Generation.* IJNLC.
- Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). *Reciprocal Rank Fusion outperforms
  Condorcet and individual rank learning methods.* SIGIR.
