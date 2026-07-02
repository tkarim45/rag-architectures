# naive — dense-RAG baseline

The canonical retrieval-augmented generation loop, implemented with no tricks:

1. Embed the question with the shared embedder.
2. Take the top-k chunks by cosine similarity from the vector store.
3. Stuff them into a numbered context block.
4. Ask the (strictly grounded) generator to answer from that context only.

## Why this package exists

**It is the baseline.** Every other architecture in this repo — hybrid, rerank, multi-query,
HyDE, GraphRAG, RAPTOR, corrective, agentic, all of them — is measured as a delta against this
package on the same corpus, chunker, embedder, and generator. If an architecture cannot beat
naive dense retrieval, its extra latency and LLM calls are not paying rent. Keeping this package
minimal (one embedding call, one ANN lookup, zero online LLM calls before generation) is what
makes those comparisons meaningful.

## What it does well

- **Paraphrase robustness.** Embeddings match by meaning, so "who started the company?" finds a
  chunk that says "founded by" — the vocabulary-mismatch problem that sinks lexical retrieval is
  handled reasonably by the embedding space itself.
- **Speed and cost.** One embedding call per query; no LLM spend until generation. This is the
  latency/cost floor for the whole repo.
- **Single-hop factoid questions.** When the answer lives in one chunk that resembles the
  question, dense top-k is very hard to beat.

## Where it fails (by design — the other packages are the fixes)

1. **Multi-hop questions are impossible in one shot.** A question like "what does the company
   founded by X sell?" needs a *bridge* document (X founded Veyra) before the answer document
   (Veyra sells ...) can even be phrased as a query. One query vector can point at only one
   region of embedding space; there is no second hop. Iterative/structural architectures
   (agentic, GraphRAG, RAPTOR) exist for exactly this.
2. **Precision decays as k grows.** Raising `top_k` to chase recall drags in near-neighbors that
   are topically similar but irrelevant, and the strictly-grounded generator then has more
   distractors to be misled by (or to abstain over). Reranking and fusion architectures exist to
   spend k on better candidates rather than more of them.

A residual weakness worth knowing: while embeddings *usually* absorb vocabulary mismatch, they
blur exact strings — rare entities, IDs, error codes. That is the sparse (BM25) package's home
turf; hybrid fuses the two.

## Usage

```python
import naive
from core import Runtime

rt = Runtime.for_testing()                # or Runtime.from_env() for real Claude + ST embeddings
pipe = naive.Pipeline(rt, naive.Config(top_k=5))

result, context = pipe.retrieve("Who founded Veyra Systems?")   # benchmark path (no generation)
print(result.doc_ids, result.diagnostics["scores"])

full = pipe.answer("Who founded Veyra Systems?")                # standalone path
print(full.answer.text)
```

## Files

| File | Role |
|---|---|
| `config.py` | `NaiveConfig` — chunker, `top_k`, context budget (frozen dataclass) |
| `retriever.py` | `DenseRetriever` — the `core.retrieval.Retriever` over `CorpusIndex.dense_search` |
| `pipeline.py` | `Pipeline` — retrieve → context → generate, fully traced |
| `ARCHITECTURE.md` | Data-flow diagram, component table, failure modes, tuning guide |
