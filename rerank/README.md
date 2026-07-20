# Rerank RAG (two-stage retrieve-then-rerank)

A **recall-then-precision funnel**: stage 1 over-fetches `candidate_k` chunks with cheap
bi-encoder search (optionally unioned with BM25), stage 2 re-scores those candidates with a
cross-encoder and keeps the top `final_k`.

## Why two stages

The two model families sit at opposite ends of a cost/quality trade:

| | Bi-encoder (stage 1) | Cross-encoder (stage 2) |
|---|---|---|
| How | embed query & passage **independently**, compare vectors | full attention over the **concatenated** query+passage |
| Cost per query | one embedding + ANN lookup (passage vectors precomputed offline) | one transformer forward pass **per candidate**, nothing precomputable |
| Quality | coarse, all interaction squeezed through one dot product | sharp, every query token attends to every passage token |

So neither works alone at corpus scale: the cross-encoder is too slow to score every chunk, and
the bi-encoder's top few are often mis-ordered. The funnel gets both, bi-encoder recall over the
whole corpus, cross-encoder precision over just `candidate_k` survivors (Nogueira & Cho 2019).

**The latency tradeoff is explicit:** rerank adds `candidate_k / batch_size` cross-encoder forward
passes per query on top of naive retrieval. You pay that only when stage-1 ordering is actually
wrong, the `rank_movement` diagnostics tell you whether it is.

**The classic failure:** `candidate_k` too small. The reranker can only *reorder* what stage 1
surfaced; if the gold chunk isn't in the candidates, no amount of reranking recovers it. Watch
`top1_from_stage1_rank` in diagnostics, if winners regularly come from deep in the candidate
list, you're near the cliff and should raise `candidate_k`.

## Usage

```python
from core import Runtime
import rerank

rt = Runtime.from_env()
pipe = rerank.Pipeline(rt)                     # lazy-loads the cross-encoder on first query
result, context = pipe.retrieve("Who founded Veyra Systems?")   # benchmark path
print(pipe.answer("Who founded Veyra Systems?").answer.text)    # standalone path

# offline / tests: inject a deterministic, dependency-free reranker
pipe = rerank.Pipeline(Runtime.for_testing(), reranker=rerank.LexicalOverlapReranker())
```

Key `Config` fields: `candidate_k` (default 20), `use_sparse_candidates`, `final_k`,
`cross_encoder_model` (default `cross-encoder/ms-marco-MiniLM-L-6-v2`), `batch_size`,
`score_threshold`. Funnel diagnostics (candidate counts, rank movement, threshold drops) land in
`RetrievalResult.diagnostics`.

`Reranker` is a Protocol, anything with `rerank(query, chunks) -> chunks` plugs in.
`LexicalOverlapReranker` (token Jaccard, deterministic, zero deps) is the offline-test double and
the documented fallback when the cross-encoder can't load; it is weaker than the cross-encoder,
not a substitute.

See `ARCHITECTURE.md` for the funnel diagram, failure modes, and tuning guide.

## Reference

Nogueira & Cho (2019). *Passage Re-ranking with BERT.* arXiv:1901.04085.
