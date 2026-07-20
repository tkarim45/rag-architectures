# HyDE: Hypothetical Document Embeddings

Zero-shot dense retrieval that sidesteps the query→document vocabulary gap: instead of embedding
the user's question, have the LLM **write the document it wishes existed**, embed *that*, and
search with it.

Paper: Gao et al. 2022, *Precise Zero-Shot Dense Retrieval without Relevance Labels*
([arXiv:2212.10496](https://arxiv.org/abs/2212.10496)).

## Why it works

Embedding models are trained mostly on text→text similarity, so mapping a short interrogative
("Who founded Veyra Systems?") near a long declarative corpus passage is a harder task than
mapping two passages near each other. HyDE converts the hard task into the easy one:

1. The LLM writes `n_hypotheses` short **hypothetical documents** that would answer the question.
   Invented facts are fine, the hypothesis is never shown to anyone; only its vocabulary and
   shape matter.
2. Each hypothesis is embedded.
3. The hypothesis vectors are averaged and blended with the *real* query vector at
   `query_weight`, then L2-renormalized.
4. The mixed vector probes the dense index (`index.dense_search_vector`); top chunks become the
   generation context.

### The `query_weight` tradeoff

The paper's InstructGPT + Contriever setup averages the query vector together with the hypothesis
vectors; we expose that as a dial. `query_weight = 0.0` is paper-pure HyDE, maximal vocabulary
transfer, but the search inherits **hallucinated-entity drift**: if the LLM invents "Nordwave
Analytics" into the hypothesis, the probe drifts toward whatever in the corpus resembles that
invention. `query_weight = 1.0` degenerates to naive dense retrieval. The default `0.25` keeps
the hypothesis dominant while the real question anchors the probe. See `ARCHITECTURE.md` for the
tuning table.

## Usage

```python
from core import Runtime
from hyde import Config, Pipeline

pipeline = Pipeline(Runtime.from_env(), Config(query_weight=0.25))

result, context = pipeline.retrieve("Who founded Veyra Systems?")
print(result.diagnostics["hypotheses"])   # the generated hypothetical documents
print(result.doc_ids)                     # ranked unique doc ids

print(pipeline.answer("Who founded Veyra Systems?").answer.text)
```

Offline (no network, no model downloads):

```python
import core

rt = core.Runtime.for_testing(llm=core.FakeLLM().on(
    "hypothetical document",
    "Veyra Systems was founded by Mara Lindqvist in Tallinn in 2014."))
result, context = Pipeline(rt).retrieve("Who founded Veyra Systems?")
```

The hypothesis prompt contains the phrase `"hypothetical document"`, so a single `FakeLLM.on()`
rule routes it deterministically.

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `n_hypotheses` | `1` | Hypothetical documents generated per question. >1 requires `temperature > 0` (enforced) or every hypothesis is identical. |
| `temperature` | `0.0` | Sampling temperature for hypothesis generation. |
| `query_weight` | `0.25` | Weight of the real query vector in the mixed probe, in [0, 1]. |
| `hypothesis_max_tokens` | `256` | Generation budget per hypothesis. |
| `top_k` | `8` | Chunks pulled from the dense index. |
| `final_k` | `5` | Passages handed to the generator. |
| `chunker` | `"sentence"` | Offline chunking strategy for the index. |
| `max_context_chars` | `6000` | Character budget for the context block. |

## Honest finding from this repo's benchmark

**HyDE scored 0% on multi-hop questions.** A hypothetical *answer* paragraph still cannot resemble
a **bridge document** that shares no vocabulary with the question, if answering requires first
finding "X works at Y" and then "Y is headquartered in Z", no single hypothesis embeds near the
intermediate document. HyDE fixes **vocabulary mismatch**, not **structural hops**; for multi-hop,
reach for iterative/structural architectures (`agentic`, `raptor`) instead.

## Files

| File | Role |
|---|---|
| `config.py` | Frozen `Config` dataclass, every tunable. |
| `prompts.py` | The hypothesis prompt (all LLM touchpoints). |
| `hypothesis.py` | n independent LLM calls → `list[str]` hypotheses. |
| `retriever.py` | Vector mixing + `dense_search_vector` + diagnostics. |
| `pipeline.py` | `Pipeline.retrieve()` / `Pipeline.answer()` per the package contract. |
| `ARCHITECTURE.md` | Data-flow diagram, failure modes, tuning guide. |
