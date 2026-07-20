# Agentic RAG (ReAct)

Retrieval as a **tool-using agent loop**. Instead of a fixed retrieve-then-generate pipeline, the
LLM holds four retrieval tools and drives its own investigation: think about what it knows, call
one tool, read the observation, and repeat, until it can answer from evidence it actually
retrieved. This is the ReAct pattern (Yao et al. 2023, [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)).

```
loop (≤ max_steps):
  LLM → {"thought", "action", "action_input"}     # structured JSON via StructuredCaller
      → run tool (search | keyword_search | read_document | list_documents)
      → Observation appended to scratchpad
until {"thought", "final_answer"} or budget exhausted
```

Every chunk a search surfaces and every document the agent reads is recorded in an **EvidenceLog**;
the pipeline ranks that evidence (touch frequency + trajectory recency) into the `RetrievalResult`
the benchmark scores. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full data flow.

## Why this architecture exists

Multi-hop questions ("who founded the company that makes the database Quorrel uses?") are
unanswerable by any single query, no rephrasing of the question embeds near the *bridge* document
(Talix → Brightfen), because the question doesn't mention it. The agent's follow-up searches are a
mechanism that genuinely chains hops: search "Quorrel database" → read that Quorrel stores state in
Talix → search/read Talix → discover Brightfen → answer.

## The honest benchmark finding

On this repo's shared benchmark (12 labeled questions, 14-doc fictional corpus, identical index
and LLM across all 13 architectures):

| architecture | recall@5 | hit-rate | answer acc | multi-hop acc (n=4) |
|---|---|---|---|---|
| **agentic** | 0.97 | 100% | **83%** | **50%** |

**Agentic was the top scorer** (tied with RAPTOR), 83% overall and 50% multi-hop, while every
query-transform method (multi-query, RAG-fusion, HyDE) scored **0% on multi-hop**. The reason is
structural, not incidental: follow-up searches informed by intermediate reads are the one mechanism
in the fixed-pipeline family's blind spot, rephrasing a question can never retrieve a bridge
document the question doesn't mention.

**What it costs, be honest about both sides:**

- **k× LLM calls per query.** Every step is one LLM round-trip; a 3-hop chain costs 3 to 5 calls
  where naive RAG costs 1. Latency and spend scale with trajectory length.
- **Non-determinism in trajectory length.** With a sampled LLM, the same question can take 2 steps
  on one run and 6 on the next (or stall and hit the budget). Fixed pipelines have fixed cost;
  this one has a cost *distribution*.

## Usage

```python
from core import Runtime
import agentic

runtime = Runtime.from_env()                    # or Runtime.for_testing() offline
pipeline = agentic.Pipeline(runtime)            # optionally: agentic.Config(max_steps=6), index=...

retrieval, context = pipeline.retrieve("Who founded the company that makes the database Quorrel uses?")
print(retrieval.doc_ids)                        # ranked evidence docs
print(retrieval.diagnostics["trajectory"])      # full thought/action/observation trace

result = pipeline.answer("Who founded the company that makes the database Quorrel uses?")
print(result.answer.text)                       # the agent's own final answer (or generator fallback)
```

`retrieve()` and `answer()` on the same question share **one** agent run (a small FIFO cache keyed
by the question), so benchmarking retrieval and then reading the answer doesn't pay for, or
diverge across, two trajectories.

## Files

| file | what it owns |
|---|---|
| `config.py` | frozen `Config`, step/output budgets, per-tool k, evidence ranking, cache size |
| `prompts.py` | system prompt + per-step decision prompt (the `Decide your next action` contract) |
| `tools.py` | `Tool` / `ToolRegistry` + the four built-in corpus tools; errors become observations |
| `evidence.py` | `EvidenceLog`, dedup'd evidence trail → frequency+recency ranked chunks/docs |
| `agent.py` | the ReAct loop: structured decisions, duplicate guard, truncation, step spans |
| `pipeline.py` | package contract (`retrieve`/`answer`), one-run cache, generator fallback |

## Reference

- Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., Cao, Y. (2023).
  *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023.
  [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
