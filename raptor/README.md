# RAPTOR

**R**ecursive **A**bstractive **P**rocessing for **T**ree-**O**rganized **R**etrieval, 
Sarthi et al. 2024, [arXiv:2401.18059](https://arxiv.org/abs/2401.18059).

Standard RAG retrieves short contiguous chunks, so any question whose evidence is *spread across
documents* has no single chunk that answers it. RAPTOR fixes this offline: it recursively
clusters document embeddings and has an LLM summarize each cluster, producing a tree whose upper
levels are progressively more abstract, **cross-document** summary nodes. Retrieval then scores
leaves and summaries *together* ("collapsed tree", the paper's better-performing variant, and
the only one implemented here), so a broad or multi-hop query can match a summary that no single
leaf could satisfy.

```
offline:  docs → embed leaves → GMM soft-cluster (k by BIC) → LLM summarize → embed → recurse
online:   embed query → score ALL nodes (all levels) → greedy select under token budget
          → per-source-doc chunk mapping → context → answer
```

## Usage

```python
import core, raptor

runtime = core.Runtime.from_env()            # or Runtime.for_testing() for offline
tree = raptor.build_tree(runtime, runtime.corpus)      # offline, build once & share
pipe = raptor.Pipeline(runtime, tree=tree)             # tree=None → lazy build from corpus

result, context = pipe.retrieve("Who founded the company that makes the database Quorrel uses?")
print(result.doc_ids)                                   # multi-doc credit from summary nodes
print(result.diagnostics["selected_nodes"])             # which levels won, at what score
print(pipe.answer("...").answer.text)
```

## Benchmark result (honest reading)

On this repo's shared labeled corpus, RAPTOR scored **83% overall and 50% on multi-hop**, 
because summary nodes carry cross-document context that single-chunk retrieval structurally
cannot: a level-1 summary of the Veyra/Brightfen cluster contains, in one retrievable unit, facts
that live in three different source documents. That is also the honest caveat, the multi-hop
gain exists *only when* the clustering happens to group the bridge documents together and the
summary preserves the linking facts. When either fails, RAPTOR degrades to naive dense retrieval
over leaves.

Where the cost went: one LLM call per cluster per level at build time (retrieval itself adds no
LLM calls over naive RAG).

## Files

| File | What it owns |
|---|---|
| `config.py` | Frozen `Config`, every tunable (BIC sweep, soft threshold, budgets) |
| `tree.py` | `RaptorNode` / `RaptorTree` dataclasses + `build_tree` offline builder |
| `clustering.py` | GMM soft clustering, k selected by BIC (UMAP deliberately skipped, see docstring) |
| `summarizer.py` | LLM cluster summarization (temperature 0, reproducible builds) |
| `prompts.py` | The single LLM prompt in the package |
| `retriever.py` | Collapsed-tree scoring, budget-greedy selection, node→chunk contract mapping |
| `pipeline.py` | `Pipeline.retrieve` / `Pipeline.answer` contract surface |

See `ARCHITECTURE.md` for the full data-flow diagram, failure modes, and tuning guide.
