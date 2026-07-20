# Hybrid RAG: architecture

Dense and sparse retrieval as parallel branches over one chunk set, joined by rank/score fusion.

## Data flow

```mermaid
flowchart TB
    subgraph Offline ["Offline (index build — once, shared)"]
        docs[Corpus documents] --> chunker[Chunker<br/>config.chunker]
        chunker --> chunks[Chunks]
        chunks --> embed[Embedder] --> vs[(Vector store<br/>FAISS/NumPy)]
        chunks --> bm25[(BM25 index)]
    end

    subgraph Online ["Online (per query)"]
        q[Question] --> dq[dense_search<br/>k = dense_k]
        q --> sq[sparse_search<br/>k = sparse_k]
        vs -.-> dq
        bm25 -.-> sq
        dq --> fuse{Fusion<br/>rrf | weighted}
        sq --> fuse
        fuse --> cut[top final_k] --> ctx[ContextBuilder] --> gen[AnswerGenerator]
        cut --> rr[RetrievalResult<br/>+ diagnostics]
    end
```

Both branches read the **same chunks**, the comparison is retrieval strategy, never corpus
coverage.

## Components

| Component | File | Responsibility |
|---|---|---|
| `Config` | `config.py` | Frozen tunables; validates at construction (`ConfigurationError`) |
| `HybridRetriever` | `retriever.py` | Runs both branches, dispatches fusion, records diagnostics |
| `Pipeline` | `pipeline.py` | Lifecycle + wiring: lazy index build, context assembly, generation |
| `core.retrieval.fusion.rrf` | core | Rank-based fusion, `Σ 1/(rrf_k + rank)` |
| `core.retrieval.fusion.weighted_fusion` | core | Normalize (minmax/zscore) then weighted sum |

## Diagnostics (in `RetrievalResult.diagnostics`)

- `fusion`, method + its settings (`rrf_k`, or `weights` + `normalization`)
- `dense.top` / `sparse.top`, each branch's full ranking (chunk id, doc id, score)
- `branch_overlap`, chunks both branches returned; `0` means fusion alone decided the order

## Failure modes

| Failure | Symptom | Cause / fix |
|---|---|---|
| One branch poisons the fusion | weighted fusion tracks whichever branch has wilder scores | Score scales incomparable, use RRF, or fix normalization |
| Branches fully agree | hybrid ≈ naive, extra latency for nothing | Corpus/queries lack exact-token vs paraphrase split; hybrid adds no value here |
| Branch k too small | correct chunk in neither branch's top-k, fusion can't recover it | Raise `dense_k`/`sparse_k`, fusion only reorders what branches surface |
| BM25 tokenization mismatch | sparse branch misses hyphenated/cased ids | Fix chunk `index_text` normalization at ingestion |
| Multi-hop questions | fused list has hop-1 docs only | Fusion is single-shot; no branch retrieves the bridge doc. Use iterative architectures (agentic, RAPTOR) |

## Tuning

| Knob | Default | Raise when | Lower when |
|---|---|---|---|
| `dense_k` / `sparse_k` | 12 / 12 | recall@final_k low, gold chunk missing from both branches | latency-bound; branches drown fusion in noise |
| `fusion` | `rrf` |, switch to `weighted` only with calibrated scores + validation data |, |
| `rrf_k` | 60 | branches are noisy, higher k flattens rank influence | you trust each branch's head strongly |
| `weights` | (0.5, 0.5) | queries are paraphrase-heavy → up-weight dense; id/jargon-heavy → up-weight sparse |, |
| `normalization` | `minmax` | outlier scores squash minmax → try `zscore` |, |
| `final_k` | 8 | downstream context budget allows more evidence | precision matters more than recall |

## Citation

- Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). **Reciprocal Rank Fusion outperforms
  Condorcet and individual Rank Learning Methods.** *SIGIR 2009*.. RRF and the `k=60` constant.
- Robertson, S., & Zaragoza, H. (2009). **The Probabilistic Relevance Framework: BM25 and Beyond.**
  *Foundations and Trends in IR.*, the sparse branch.
