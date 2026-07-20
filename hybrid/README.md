# Hybrid RAG (dense + BM25, fused)

Dense vector search and BM25 keyword search run **in parallel over the same chunks**, and their
rankings are fused into one list. Reciprocal Rank Fusion (RRF) by default, weighted score fusion
by config.

## Why hybrid

The two branches fail in complementary ways:

| | Dense (bi-encoder cosine) | Sparse (BM25) |
|---|---|---|
| Strength | paraphrase, synonyms, intent | exact tokens: names, ids, error codes, jargon |
| Blind spot | rare/unseen exact terms | rewordings, vocabulary mismatch |

Hybrid's win condition is *disagreement*: when a query contains both a rare exact token and a
paraphrased intent, each branch recovers the half the other misses, and fusion promotes the chunks
both agree on.

## Why RRF is the default

BM25 scores are unbounded term-frequency sums; cosine similarities live in [-1, 1]. The two live on
**incomparable scales**, so any direct score arithmetic silently lets one branch dominate. RRF
(Cormack et al. 2009) is rank-based and therefore scale-free, it only asks "where did each branch
place this chunk?", scoring `Σ 1/(rrf_k + rank)`. One hyperparameter (`rrf_k=60`, the canonical
constant), no normalization to tune, robust out of the box.

**When weighted fusion wins:** if your branch scores are calibrated (or you normalize them, 
`minmax`/`zscore` supported) and you have validation data to tune the `(dense, sparse)` weights,
weighted fusion preserves score-*magnitude* information RRF throws away and can edge it out. It is
the tuned option, not the safe one.

## Usage

```python
from core import Runtime
import hybrid

rt = Runtime.from_env()                       # or Runtime.for_testing() for offline
pipe = hybrid.Pipeline(rt)                    # or Pipeline(rt, hybrid.Config(fusion="weighted"))
result, context = pipe.retrieve("Who founded Veyra Systems?")   # benchmark path
print(pipe.answer("Who founded Veyra Systems?").answer.text)    # standalone path
```

Key `Config` fields: `dense_k` / `sparse_k` (per-branch fan-out), `fusion` (`"rrf"` | `"weighted"`),
`rrf_k`, `weights`, `normalization`, `final_k`, `chunker`, context budget. Per-branch rankings,
overlap, and the fusion settings are recorded in `RetrievalResult.diagnostics`.

See `ARCHITECTURE.md` for the data-flow diagram, failure modes, and tuning guide.

## Reference

Cormack, Clarke & Buettcher (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual
Rank Learning Methods.* SIGIR 2009.
