# adaptive: Adaptive-RAG (complexity-routed retrieval)

Implementation of **Adaptive-RAG** (Jeong et al. 2024, *Adaptive-RAG: Learning to Adapt
Retrieval-Augmented Large Language Models through Question Complexity*, arXiv:2403.14403):
classify each question's complexity first, then run only the cheapest retrieval strategy
predicted to be sufficient.

```
question → complexity classifier ─┬─ A  no_retrieval  → empty context (generator abstains)
                                  ├─ B  single_step   → one dense top-k pass
                                  └─ C  multi_step    → iterative loop:
                                        evidence ← fused(dense+BM25, RRF) on question
                                        repeat: "done, or what additional information?"
                                                → follow-up query → fuse → append new chunks
                                        until done | stalled | max_iterations
```

## The bet: and the risk

**The bet:** query complexity is wildly skewed. Most real questions are single-hop ("who founded
Veyra Systems?") and are answered perfectly well by one cheap dense pass; only a minority need
fact-chaining ("who founded the company that makes the database Quorrel uses?"). A one-size
pipeline must either overpay on every easy query (always-iterate) or fail every hard one
(always-single-shot). Routing buys multi-hop capability at near-single-shot average cost.

**The risk, honestly:** *the classifier IS the architecture.* Every route is a known quantity, 
B is literally the naive baseline, C is a standard iterative chain. All this package adds is the
decision between them, so its error budget is the classifier's error budget:

- **C → B misroute is the killing one.** A multi-hop question routed to single-step lands on
  single-shot dense retrieval, which structurally *cannot* fetch the bridge document (the second
  hop's query can't even be phrased until the first hop is read). The result is a confident
  recall-zero, not a graceful degradation.
- **B → C misroute merely wastes money**, the loop's first decision usually says "done" and the
  answer is unchanged. This asymmetry is why the classifier falls back to **C**, not B, when its
  output is unusable, and why "route accuracy" matters more on hard queries than easy ones.

Prior data point from this repo: an earlier incarnation of this package, a cruder three-way
router (substring-matched free-text label, no structured output, no iterative route of its own)
,  scored **58% overall and 25% on multi-hop** in the benchmark. Multi-hop is exactly where the
router's errors concentrate, and exactly what this rewrite's structured classifier + genuine
iterative C route are aimed at.

## Why `allow_no_retrieval` defaults to False

The paper's A route ("answer from parametric memory, skip retrieval") is a real cost win in
open-domain settings, no lookup for "what is the capital of France?". **On this repo's corpus it
can only ever be wrong**: the corpus is closed and fictional by construction (Veyra, Quorrel,
Brightfen…), so no model has these facts in pretraining and a parametric answer is hallucination
by definition. Route A is therefore implemented but disabled by default: the classifier's `A` is
coerced to `B` (recorded as `coerced: true` in diagnostics), and if you enable it the strictly
grounded core generator receives an empty context and abstains honestly. The route exists because
the architecture is general; the default exists because this corpus is adversarial to it.

## What the multi-step route actually does (paper's C)

1. **Seed** the evidence pool: dense + BM25 retrieval on the original question, fused with RRF.
2. **Decide**: show the LLM the question + accumulated evidence, ask (structured JSON) whether
   the evidence is sufficient, and if not, *what additional information* to retrieve next.
3. **Retrieve** the follow-up query with the same fusion; append only chunks not already held.
4. Stop on `done`, on **stall** (a follow-up added zero new chunks, iterating again would show
   the LLM the identical evidence and, at temperature 0, produce the identical decision), or at
   `max_iterations`. The stop reason is always recorded.

Fusion (rather than dense-only) on this route is deliberate: follow-up queries name exact
entities learned from evidence ("Brightfen founder"), and exact names are BM25's strength.

## Usage

```python
import adaptive
from core import Runtime

rt = Runtime.from_env()                      # or Runtime.for_testing() offline
pipe = adaptive.Pipeline(rt, adaptive.Config(max_iterations=3))

result, context = pipe.retrieve("Who founded the company that makes the database Quorrel uses?")
print(result.diagnostics["route"])           # "C"
print(result.diagnostics["iterations"])      # per-iteration sub-queries + newly found docs
print(result.diagnostics["stop_reason"])     # classifier_done | stalled | max_iterations

full = pipe.answer("Who founded Veyra Systems?")   # routed "B": one dense pass, then generate
print(full.answer.text)
```

## Diagnostics you get per query

| Key | Meaning |
|---|---|
| `route` / `route_name` | Effective label (`A`/`B`/`C`) and the strategy that ran |
| `classifier` | `raw_label`, `reason`, `coerced` (A→B policy), `fallback` (unusable output → C) |
| `sub_queries` | Follow-up queries the multi-step loop issued (also on `query.variants`) |
| `seed` / `iterations` | Seed hits; per-iteration decision, next_query, and newly found chunk/doc ids |
| `stop_reason` | `classifier_done` \| `stalled` \| `max_iterations` \| `decision_error` \| `no_seed_evidence` |
| `evidence_chunks` / `kept_chunks` | Accumulated pool size vs what survived the `final_k` cap |

## Files

| File | Role |
|---|---|
| `config.py` | `AdaptiveConfig`, routing policy, per-route budgets, context budget (frozen) |
| `prompts.py` | The two LLM touchpoints: complexity classifier + follow-up decision |
| `classifier.py` | `ComplexityClassifier`, structured A/B/C call, A→B coercion, C fallback |
| `strategies.py` | The three routes: `no_retrieval`, `single_step`, `MultiStepRetriever` |
| `retriever.py` | `AdaptiveRetriever`, classify → dispatch → one merged diagnostics story |
| `pipeline.py` | `Pipeline`, retrieve → context → grounded answer, fully traced |
| `ARCHITECTURE.md` | Data-flow diagram, component table, failure modes, tuning guide |
