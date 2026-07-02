# sparse — BM25 lexical retrieval

Okapi BM25 over the corpus chunks: tokenize → stopword-filter → (optionally) stem → score query
terms against an inverted index. The retrieval half is lexical; everything downstream (context
assembly, grounded generation) is the same shared core machinery as `naive`, so benchmark deltas
between the two packages measure exactly one thing: lexical vs semantic matching.

## Why BM25 still matters

Twenty-five years on, BM25 remains the retrieval baseline that embedding models are published
against — and the one they still lose to on a specific, important query class:

- **Rare entities and exact strings.** Product codes, error IDs, person names, `VYR-2041`-style
  identifiers. Embeddings blur surface forms into meaning; BM25's inverted index treats a rare
  term as gold (high IDF) and retrieves precisely the chunks containing it.
- **Zero online model cost.** No embedding call, no GPU; a tokenize and an index scan. This is
  the cheapest online path in the repo.
- **Explainability.** A BM25 score decomposes into per-term contributions. When retrieval goes
  wrong, `diagnostics["query_terms"]` shows exactly which terms were searched — there is no
  embedding-space mystery to debug.

## Where it fails

**Vocabulary mismatch is structural.** BM25 scores literal term overlap; if the question says
"who started the company" and the chunk says "founded by", the overlap is zero and the score is
zero. No k1/b tuning fixes this — it is the reason dense retrieval exists, and why `hybrid/`
fuses the two. Expect this package to lose to `naive` on paraphrased questions and beat it on
entity/ID lookups.

**Multi-hop is equally impossible** as it is for naive: one bag of query terms, one shot.

## The analyzer is where quality lives

The BM25 *formula* is two parameters; the **analyzer** (tokenization, stopwords, stemming,
minimum token length) decides what a "term" even is, and that is where real-world BM25 systems
are actually won or lost. This package therefore:

- exposes the full analyzer in `SparseConfig` (`stem`, `min_token_len`, `extra_stopwords`);
- guarantees query-side and index-side analysis are the *same object* — analyzer asymmetry is the
  classic silent BM25 bug and is unrepresentable here;
- **respects the config**: when your `k1`/`b`/analyzer settings differ from the core defaults,
  `BM25Retriever` builds its own `core.stores.lexical.BM25Index` over the shared index's chunks
  instead of silently answering from the default-parameter index (which would make every tuning
  experiment a no-op). `diagnostics["custom_bm25_index"]` records which path was taken.

## Usage

```python
import sparse
from core import Runtime

rt = Runtime.for_testing()
pipe = sparse.Pipeline(rt, sparse.Config(k1=1.2, b=0.6, extra_stopwords=("company",)))

result, context = pipe.retrieve("Who founded Veyra Systems?")   # benchmark path
print(result.doc_ids, result.diagnostics["query_terms"])

full = pipe.answer("Who founded Veyra Systems?")                # standalone path
print(full.answer.text)
```

## Files

| File | Role |
|---|---|
| `config.py` | `SparseConfig` — k1/b, analyzer knobs, top_k, context budget (frozen dataclass) |
| `retriever.py` | `BM25Retriever` — reuses the shared index's BM25 at default config, builds its own when tuned |
| `pipeline.py` | `Pipeline` — retrieve → context → generate, fully traced |
| `ARCHITECTURE.md` | Data-flow diagram, component table, failure modes, tuning guide, citation |
