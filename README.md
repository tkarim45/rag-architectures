# rag-architectures

**13 RAG architectures, one corpus, one labeled eval set — measured side by side on real
infrastructure.** Each architecture is its own production-style package (config · retriever ·
pipeline · architecture-specific modules); they all run over the same documents and embeddings, so
the comparison is honest. Retrieval is scored on recall, answers are graded by an LLM judge, and the
whole thing runs on **real Claude (AWS Bedrock) + real sentence-transformer embeddings + a FAISS
vector store** — no mock.

```bash
cp .env.example .env          # add AWS Bedrock creds (see "Running" below)
rag-bench                     # run every architecture over the eval set, print the table
rag-bench --methods graphrag,agentic --limit 4
```

## The architectures

Each is a self-contained folder with its own code + README:

| family | folder | one-line |
|---|---|---|
| **Retrieval** | [`naive/`](naive) | dense top-k, one pass (the baseline) |
| | [`sparse/`](sparse) | BM25 lexical |
| | [`hybrid/`](hybrid) | dense + BM25 fused with RRF |
| | [`rerank/`](rerank) | dense recall → cross-encoder precision |
| **Query transform** | [`multi_query/`](multi_query) | LLM rephrasings → union |
| | [`rag_fusion/`](rag_fusion) | rephrasings → RRF |
| | [`hyde/`](hyde) | embed a hypothetical answer |
| **Indexing** | [`chunking/`](chunking) | sentence-window / parent-child / contextual |
| **Structural** | [`graphrag/`](graphrag) | entity graph + multi-hop traversal |
| | [`raptor/`](raptor) | cluster + summarize tree |
| **Control-flow** | [`corrective/`](corrective) | grade docs → rewrite + re-retrieve (CRAG) |
| | [`adaptive/`](adaptive) | router → dense / graph / fusion |
| | [`agentic/`](agentic) | tool-using retrieval agent loop |

Shared infra lives in [`common/`](common) — corpus, embeddings + Bedrock client, the chunkers +
FAISS index, generation, scoring. (This is shared on purpose: every architecture must run on the
*same* corpus and embedding model or the benchmark isn't comparable.)

## The eval

A **fully fictional, interlinked knowledge base** (14 docs about invented companies/people/products)
+ **12 labeled questions** — single-hop and multi-hop — each with its gold documents and a reference
answer. Fictional on purpose: the answers can't be in any model's training data, so a method only
scores if it actually *retrieves* the right docs (this measures RAG, not memorization). The
multi-hop questions ("who founded the company behind the database Quorrel uses?") require chaining
across documents — which is where the architectures separate.

## Measured results

One real Bedrock run (Claude Haiku 4.5, FAISS exact search), 12 questions / 14 docs, recall@5:

| architecture | recall@5 | hit-rate | answer acc | **multi-hop acc** (n=4) |
|---|---|---|---|---|
| **agentic** | 0.97 | 100% | **83%** | **50%** |
| **raptor** | 0.92 | 100% | **83%** | **50%** |
| hybrid | 0.94 | 100% | 75% | 50% |
| chunk:sentence_window | 0.94 | 100% | 75% | 50% |
| chunk:parent_child | 0.94 | 100% | 75% | 50% |
| rerank | 0.92 | 100% | 67% | 25% |
| graphrag | 0.86 | 100% | 67% | 25% |
| naive | 0.94 | 100% | 58% | **0%** |
| sparse | 0.92 | 100% | 58% | 25% |
| chunk:contextual | 0.86 | 100% | 58% | 25% |
| adaptive | 0.86 | 100% | 58% | 25% |
| multi_query | 0.89 | 100% | 50% | **0%** |
| rag_fusion | 0.92 | 100% | 50% | **0%** |
| hyde | 0.86 | 100% | 50% | **0%** |
| corrective | 0.89 | 100% | 50% | **0%** |

### The honest finding: multi-hop is the discriminator, and rephrasing doesn't solve it

Hit-rate is 100% everywhere (with 14 docs, the gold doc is almost always *somewhere* in the top-5),
so **answer accuracy on multi-hop questions** is what actually separates the architectures:

- **Query-transform methods score 0% on multi-hop** — `multi_query`, `rag_fusion`, `hyde`, and the
  `naive` baseline. No rephrasing of "who founded the company behind the database X uses" lexically
  or semantically resembles the *founder* document; you can reword the question forever and never
  retrieve the bridge doc. Multi-hop needs **structure or iteration**, not lexical breadth — the
  same structure-beats-vocabulary lesson as a good router.
- **The methods that gain on multi-hop add a second mechanism:** an agent that issues follow-up
  searches (`agentic`), a summary tree that carries cross-doc context (`raptor`), or simply more
  context per hit (`hybrid`, parent-child chunking). They reach 50%.
- **GraphRAG underperformed its promise here** (67%, 25% multi-hop). Its traversal *does* reach
  bridge docs, but at this scale it's gated by LLM entity-extraction quality and a 2-hop limit, so
  3-hop chains slip through. An honest result, not a cherry-picked win.

> ⚠️ **Read these as directional, not a leaderboard.** 12 questions + an LLM-as-judge means a single
> question flipping swings a method ~8 points, and Bedrock generation/judging isn't perfectly
> deterministic even at temperature 0 — across runs, methods move ±1 question. The *robust* signal
> is the structural one (query-transform = 0% multi-hop; structural/iterative > 0%), not the exact
> ranking. A production evaluation would use hundreds of questions and multiple judge samples.

## Architecture

```
common/                       # shared, so the benchmark is comparable
  corpus.py · providers.py    # fictional KB + labeled eval ; embeddings (local) + Claude (Bedrock)
  index.py · vectorstore.py   # chunkers + BM25 ; pluggable FAISS/NumPy dense store
  retrieval.py · transform.py · generate.py · evaluate.py
<architecture>/               # one folder each, prod-style
  config.py                   # tunables
  retriever.py (+ extractor/graph_builder/traversal/clustering/summarizer/tree/grader/router/tools/agent…)
  pipeline.py                 # Pipeline.answer() (standalone) + run() (benchmark adapter)
  __init__.py · README.md
benchmark.py · cli.py
```

The research-standard split is reflected in the code: **offline indexing** (build embeddings → write
into the vector store, build the graph/tree once) is separate from **online retrieval** (the
`retriever`/`pipeline` path that serves a query).

### Vector store

Dense retrieval goes through a pluggable **`VectorStore`** (`common/vectorstore.py`): **FAISS**
`IndexFlatIP` by default (exact — inner product == cosine on L2-normalized embeddings), with a NumPy
fallback (identical results). At 14 docs a flat index is exact and optimal; the same `add`/`search`
interface is the seam where you'd drop in HNSW/IVF or a hosted store (Chroma / Qdrant / pgvector) for
a real corpus. Select with `RAGARCH_VECTORSTORE=faiss|numpy`.

## Running

Real-only — there is no mock. You need **AWS Bedrock access** (every LLM call — generation, entity
extraction, summaries, grading, routing, the agent loop, HyDE, query-gen, the judge — runs on Claude
via Bedrock). Embeddings + the cross-encoder run locally (no key, one-time model download).

```bash
pip install -e ".[dev]"
cp .env.example .env          # AWS creds + region (or use a profile / Bedrock API key)
rag-bench                     # full run; a complete run is ~600–800 Bedrock calls on Haiku (~$1–3)
rag-bench --methods naive,graphrag,agentic --limit 4   # cheaper subset while iterating
```

`.env` is gitignored; the project also reads the global `~/.env`. See `.env.example` for the exact
variables.

## Tests

```bash
pytest -q          # 6 passed — pure-logic (corpus, chunkers, RRF, recall, graph traversal)
```

Tests monkeypatch the LLM/embeddings, so **CI runs offline with no Bedrock calls**. The real
embedding + Bedrock behavior is validated by an actual `rag-bench` run (the table above), not in CI.

## Stack

sentence-transformers (MiniLM embeddings + cross-encoder reranker), FAISS (vector store), rank-bm25
(sparse), scikit-learn (RAPTOR clustering), networkx (GraphRAG), Claude on AWS Bedrock
(`anthropic[bedrock]`) for all generation/reasoning, pytest.

## License

MIT
