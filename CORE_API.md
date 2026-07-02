# Core framework contract (for architecture packages)

Every architecture package builds on `core/` and **only** `core/`. No architecture package may
import another architecture package — each one implements its full retrieval strategy from core
primitives so the packages stay independently readable, testable, and deletable.

## The runtime (dependency injection)

```python
from core import Runtime

runtime = Runtime.from_env()       # real: Claude (Bedrock/Anthropic) + sentence-transformers
runtime = Runtime.for_testing()    # offline: FakeLLM + HashingEmbedder + NumPy store

runtime.llm                        # core.llm.LLM — .complete(CompletionRequest) -> Completion
runtime.embedder                   # core.embeddings.Embedder — .embed_texts / .embed_query / .dim
runtime.tracer                     # core.telemetry.Tracer — with tracer.span("name", k=v) as s: ...
runtime.corpus                     # list[Document] — the shared fictional KB
runtime.build_index("sentence")    # -> CorpusIndex (chunk → embed → FAISS + BM25)
```

Chunker names: `whole | sentence | fixed | sentence_window | parent_child | contextual`.

## Domain types (core.types)

- `Document(doc_id, title, text, metadata)`
- `Chunk(chunk_id, doc_id, index_text, display_text, metadata)` — chunk ids are `"{doc_id}::spec"`
- `Query(text, top_k, variants, metadata)`
- `ScoredChunk(chunk, score, retriever)` — has `.chunk_id`, `.doc_id`
- `RetrievalResult(query, chunks: list[ScoredChunk], diagnostics: dict)` — `.doc_ids` = ranked
  unique doc ids (what metrics score); put your architecture's story (routes, grades, hops,
  generated queries) into `diagnostics`
- `ContextBlock(text, chunk_ids, doc_ids, truncated)`
- `GeneratedAnswer(text, citations, usage, abstained)`
- `PipelineResult(query, retrieval, context, answer, diagnostics)`

## Key services

```python
from core import (ContextBuilder, AnswerGenerator, StructuredCaller, rrf, weighted_fusion,
                  CompletionRequest)

index.dense_search(query_text, k)          # -> list[ScoredChunk]
index.dense_search_vector(vec, k)          # for HyDE-style custom query vectors
index.sparse_search(query_text, k)         # BM25 -> list[ScoredChunk]
index.chunk(cid) / index.document(doc_id) / index.chunks_of(doc_id) / index.documents

rrf([ranking_a, ranking_b], k=60)          # rank fusion -> list[ScoredChunk]
weighted_fusion([a, b], [0.5, 0.5], normalization="minmax")

ContextBuilder(max_passages=5, max_chars=6000).build(chunks)  # -> ContextBlock
AnswerGenerator(runtime.llm).generate(question, context_block) # -> GeneratedAnswer

StructuredCaller(runtime.llm).call(prompt, validator=fn)      # JSON out, parse-retry-repair
runtime.llm.complete_text(prompt, max_tokens=64)               # plain text convenience
```

Errors: raise/propagate `core.errors` types. Telemetry: wrap every pipeline stage in
`runtime.tracer.span(...)`.

## The package contract

Every architecture package exposes from its `__init__.py`:

```python
from .config import <Arch>Config          # frozen dataclass of tunables, sensible defaults
from .pipeline import Pipeline
```

`Pipeline` must provide:

```python
class Pipeline:
    def __init__(self, runtime: Runtime, config: <Arch>Config | None = None, *,
                 index: CorpusIndex | None = None, ...):    # extra prebuilt resources allowed
        """`index` (and graph/tree where applicable) can be injected by the benchmark so all
        architectures share identical offline artifacts; when omitted, build lazily from
        runtime.corpus on first use."""

    def retrieve(self, question: str) -> tuple[RetrievalResult, ContextBlock]:
        """Online retrieval path only — no answer generation. This is what the benchmark calls."""

    def answer(self, question: str) -> PipelineResult:
        """retrieve() + AnswerGenerator — the standalone entrypoint."""
```

Architectures with heavyweight offline artifacts also expose a module-level builder the benchmark
can call once and share, e.g. `graphrag.build_graph(runtime, documents, config) -> KnowledgeGraph`,
`raptor.build_tree(runtime, documents, config) -> RaptorTree`.

## Quality bar

- Full type hints; docstrings explain *design decisions*, not syntax.
- Config over constants: every tunable lives in the package `Config`.
- All LLM touchpoints in a `prompts.py`; structured LLM outputs go through `StructuredCaller`.
- Must import and run **offline** under `Runtime.for_testing()` with a `FakeLLM` configured via
  `.on(substring, response)` rules.
- `ARCHITECTURE.md` per package: mermaid diagram of the data flow (offline vs online split),
  component table, failure modes, tuning guide, paper citation.
