# Corrective RAG (CRAG)

Don't trust the first retrieval — **grade it, then act on the grade**.

Implementation of Yan et al. 2024, *Corrective Retrieval Augmented Generation*
([arXiv:2401.15884](https://arxiv.org/abs/2401.15884)), on the shared `core` framework. A
retrieval evaluator grades each retrieved passage (`correct` / `incorrect` / `ambiguous` with a
confidence), the grades aggregate to a per-query verdict, and the verdict selects one of three
corrective actions before anything reaches the generator.

```
dense top-k ──► grade EACH passage ──► verdict ──┬─ CORRECT   → refine passages (strip filter)
                                                 ├─ INCORRECT → rewrite query → broadened
                                                 │              hybrid fallback → refine
                                                 └─ AMBIGUOUS → refined originals ∪ fallback (RRF)
```

## Usage

```python
from core import Runtime
from corrective import Config, Pipeline

runtime = Runtime.from_env()                     # or Runtime.for_testing() offline
pipeline = Pipeline(runtime, Config())           # index built lazily; or inject index=...

result, context = pipeline.retrieve("Who founded Veyra Systems?")   # benchmark path
print(result.diagnostics["verdict"], result.diagnostics["action"])

pipeline_result = pipeline.answer("Who founded Veyra Systems?")     # standalone path
print(pipeline_result.answer.text)
```

## What each action does

| Verdict | Action (`diagnostics["action"]`) | Behavior |
|---|---|---|
| CORRECT — ≥1 passage confidently correct | `refine` | Keep the non-incorrect passages; **knowledge refinement**: split each into sentence strips, grade each strip relevant/irrelevant (cheap YES/NO call), recompose the kept strips in order. |
| INCORRECT — all passages confidently incorrect | `fallback` | Discard the retrieval. **Rewrite the query** into keyword form, then run a broadened hybrid sweep (dense ∪ BM25 over the rewrite, RRF-fused, wider `fallback_k`), then refine. |
| AMBIGUOUS — anything else | `combine` | Hedge: RRF-fuse the kept original passages with the fallback results, take the top, refine. |

**Web-search stand-in.** The paper's INCORRECT action is web search — leave the failing source
and look wider. This benchmark is a closed corpus, so the stand-in (implemented from `core`
primitives only) is the broadened dense ∪ BM25 sweep over the rewritten query. Same intent,
honest ceiling: it can only find what the corpus already contains.

**Refined provenance.** Refinement replaces a passage's display text, so the retriever
synthesizes a new chunk per refined passage: `chunk_id = f"{original}::refined"`, same
`doc_id`, `display_text` = kept strips joined. Because chunk ids are `"{doc_id}::spec"`,
citations and `RetrievalResult.doc_ids` (what the benchmark scores) still resolve to the true
source documents; the original chunk id is preserved in the refined chunk's metadata.

## Diagnostics

`RetrievalResult.diagnostics` carries the whole story: per-passage `grades` (chunk id, doc id,
grade, confidence), the aggregated `verdict`, the `action` taken, `rewritten_query` (or
`None`), `initial_chunk_ids`, `fallback_chunk_ids`, and a `refinement` report
(applied/reason/strips kept/strips dropped).

## Honest results

In the last benchmark run, corrective scored **50% overall and 0% on multi-hop**. The grading
step genuinely catches *bad* retrieval — that's where the precision wins come from — but the
corrective action is still a query rewrite, and a rewrite of the original question cannot
surface a **bridge document** whose vocabulary appears in neither the question nor its rewrite.
That is the same query-transform ceiling that multi-query / RAG-fusion / HyDE hit on this
corpus: CRAG helps **precision, not multi-hop**. Reaching bridge docs takes structural or
iterative retrieval (GraphRAG, RAPTOR, agentic), not better phrasing of a single hop.

## Files

| File | Role |
|---|---|
| `config.py` | Frozen `Config` dataclass — every tunable, validated. |
| `prompts.py` | All three LLM prompts (grade / strip relevance / rewrite). |
| `evaluator.py` | Retrieval evaluator: per-passage structured grading + verdict aggregation. |
| `refiner.py` | Knowledge refinement: decompose-then-recompose strip filtering. |
| `rewriter.py` | Query rewriter for the fallback search. |
| `retriever.py` | Orchestrates retrieve → grade → act; owns the fallback hybrid sweep. |
| `pipeline.py` | Public `Pipeline` (contract: `retrieve` / `answer`, injectable index). |

See `ARCHITECTURE.md` for the control-flow diagram, failure modes and tuning guide.
