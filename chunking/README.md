# Chunking-strategy RAG

**Retrieval quality is often decided at index time, not query time.** This package holds the
online path constant — plain dense retrieval, identical to naive RAG — and varies only how the
corpus was chunked when the index was built. Same retriever, same corpus, same queries; the only
moving part is the index-time strategy. Whatever swings, chunking did it.

## The seam every strategy exploits

A core `Chunk` carries two texts: `index_text` (what gets embedded / BM25-indexed → match
precision) and `display_text` (what the generator reads on a hit → answer context). Naive
chunking sets them equal. The three benchmarked strategies drive them apart — **match small,
return big**:

| Strategy | Indexed | Returned | The bet |
|---|---|---|---|
| `sentence_window` | one sentence | sentence ± N neighbors | precise match, paragraph context |
| `parent_child` | one sentence | the whole parent document | precise match, whole-doc context |
| `contextual` | LLM context line + sentence | the bare sentence | disambiguate the match itself |

`STRATEGY_PROFILES` in [`strategies.py`](strategies.py) profiles all six core strategies
(including the coupled baselines `whole`, `sentence`, `fixed`) — what each indexes, what it
returns, when it wins, when it loses. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the data flow,
failure modes, and tuning guide.

## Usage

```python
from core import Runtime
from chunking import Config, Pipeline, STRATEGIES

runtime = Runtime.from_env()          # or Runtime.for_testing() for offline

# one pipeline per strategy — the benchmark's loop
for strategy in STRATEGIES:           # ("sentence_window", "parent_child", "contextual")
    pipe = Pipeline(runtime, strategy=strategy)
    retrieval, context = pipe.retrieve("Who founded Veyra Systems?")
    print(strategy, retrieval.doc_ids, retrieval.diagnostics["n_index_chunks"])

# standalone: retrieve + grounded answer
result = Pipeline(runtime, Config(top_k=10, final_k=5), strategy="parent_child") \
    .answer("Who founded Veyra Systems?")
print(result.answer.text)
```

The benchmark injects pre-built indexes (`Pipeline(runtime, index=idx, strategy=name)`) so every
strategy shares identical embedder/LLM wiring; standalone usage builds lazily via
`runtime.build_index(strategy)` on first query.

## What the diagnostics show

`RetrievalResult.diagnostics` records the strategy name, index granularity (`n_index_chunks`
over `n_documents`), and — for the top hits — the matched `index_text` beside the returned
`display_text` with their size ratio (`expansion`: 1.0 = coupled, ≫1 = match-small/return-big).
That makes each strategy's mechanics inspectable per query, not just its aggregate score.

## Files

| File | Role |
|---|---|
| `config.py` | Frozen `Config`: query-path knobs (held constant) + per-strategy build knobs |
| `strategies.py` | `StrategyProfile` registry — the design card for all six core strategies |
| `retriever.py` | Plain dense retrieval + match-vs-display diagnostics |
| `pipeline.py` | `Pipeline`: lazy/injected index, `retrieve()` (benchmark) and `answer()` |
