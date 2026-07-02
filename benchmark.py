"""The benchmark harness: every architecture, one corpus, one labeled eval set, honest numbers.

Fairness contract:
  * All architectures share the SAME runtime (one embedder, one LLM) and, where applicable, the
    same pre-built offline artifacts (indexes / knowledge graph / RAPTOR tree) — so differences in
    scores are differences in *architecture*, not in infrastructure.
  * Retrieval and generation are scored separately: `Pipeline.retrieve()` gives the ranked docs and
    the context; a shared `AnswerGenerator` turns each method's context into an answer; a shared
    `CorrectnessJudge` grades it. No architecture gets a private generator prompt.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from core import (AnswerGenerator, CorpusIndex, CorrectnessJudge, Runtime, hit_at_k, mrr,
                  ndcg_at_k, questions as corpus_questions, recall_at_k)
from core.telemetry import logger


@dataclass
class SharedResources:
    """Offline artifacts built once and injected into every pipeline that needs them."""

    runtime: Runtime
    indexes: dict[str, CorpusIndex] = field(default_factory=dict)
    graph: Any = None            # graphrag.KnowledgeGraph
    tree: Any = None             # raptor.RaptorTree


CHUNK_VARIANTS = ("sentence_window", "parent_child", "contextual")


def build_resources(runtime: Runtime | None = None, *, with_graph: bool = True,
                    with_raptor: bool = True, with_chunk_variants: bool = True) -> SharedResources:
    runtime = runtime or Runtime.from_env()
    resources = SharedResources(runtime=runtime)

    index_names = ["sentence", "whole"] + (list(CHUNK_VARIANTS) if with_chunk_variants else [])
    for name in index_names:
        logger.info("building index: %s", name)
        resources.indexes[name] = runtime.build_index(name)

    if with_graph:
        import graphrag

        logger.info("building knowledge graph")
        resources.graph = graphrag.build_graph(runtime, runtime.corpus)
    if with_raptor:
        import raptor

        logger.info("building RAPTOR tree")
        resources.tree = raptor.build_tree(runtime, runtime.corpus)
    return resources


PipelineFactory = Callable[[SharedResources], Any]


def method_registry(resources: SharedResources) -> dict[str, PipelineFactory]:
    """Method name → pipeline factory. Factories are lazy so `--methods naive` never builds
    anything the subset doesn't need."""
    registry: dict[str, PipelineFactory] = {}

    _register = registry.__setitem__

    _register("naive", lambda r: __import__("naive").Pipeline(
        r.runtime, index=r.indexes["sentence"]))
    _register("sparse", lambda r: __import__("sparse").Pipeline(
        r.runtime, index=r.indexes["sentence"]))
    _register("hybrid", lambda r: __import__("hybrid").Pipeline(
        r.runtime, index=r.indexes["sentence"]))
    _register("rerank", lambda r: __import__("rerank").Pipeline(
        r.runtime, index=r.indexes["sentence"]))
    _register("multi_query", lambda r: __import__("multi_query").Pipeline(
        r.runtime, index=r.indexes["sentence"]))
    _register("rag_fusion", lambda r: __import__("rag_fusion").Pipeline(
        r.runtime, index=r.indexes["sentence"]))
    _register("hyde", lambda r: __import__("hyde").Pipeline(
        r.runtime, index=r.indexes["sentence"]))
    _register("corrective", lambda r: __import__("corrective").Pipeline(
        r.runtime, index=r.indexes["sentence"]))
    _register("adaptive", lambda r: __import__("adaptive").Pipeline(
        r.runtime, index=r.indexes["sentence"]))
    _register("agentic", lambda r: __import__("agentic").Pipeline(
        r.runtime, index=r.indexes["sentence"]))

    for variant in CHUNK_VARIANTS:
        if variant in resources.indexes:
            _register(f"chunk:{variant}", lambda r, v=variant: __import__("chunking").Pipeline(
                r.runtime, index=r.indexes[v], strategy=v))

    if resources.graph is not None:
        _register("graphrag", lambda r: __import__("graphrag").Pipeline(
            r.runtime, index=r.indexes["whole"], graph=r.graph))
    if resources.tree is not None:
        _register("raptor", lambda r: __import__("raptor").Pipeline(r.runtime, tree=r.tree))

    return registry


def run(method_names: list[str] | None = None, limit: int | None = None, k: int = 5,
        resources: SharedResources | None = None, judge_samples: int = 1) -> dict:
    resources = resources or build_resources()
    runtime = resources.runtime
    registry = method_registry(resources)
    names = method_names or list(registry)
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise KeyError(f"unknown methods {unknown}; available: {sorted(registry)}")

    eval_questions = corpus_questions()
    if limit:
        eval_questions = eval_questions[:limit]

    generator = AnswerGenerator(runtime.llm)
    judge = CorrectnessJudge(runtime.llm, samples=judge_samples)

    results: dict[str, dict] = {}
    for name in names:
        pipeline = registry[name](resources)
        totals = {"recall": 0.0, "hit": 0.0, "mrr": 0.0, "ndcg": 0.0, "correct": 0.0,
                  "latency_ms": 0.0}
        multi_hop = {"n": 0, "correct": 0}
        per_question = []
        for question in eval_questions:
            started = time.perf_counter()
            with runtime.tracer.span("benchmark", method=name, qid=question.qid):
                retrieval, context = pipeline.retrieve(question.question)
                answer = generator.generate(question.question, context)
                correct = judge.is_correct(question.question, answer.text, question.answer)
            latency_ms = (time.perf_counter() - started) * 1000.0

            ranked = retrieval.doc_ids
            recall = recall_at_k(ranked, question.gold_docs, k)
            totals["recall"] += recall
            totals["hit"] += float(hit_at_k(ranked, question.gold_docs, k))
            totals["mrr"] += mrr(ranked, question.gold_docs)
            totals["ndcg"] += ndcg_at_k(ranked, question.gold_docs, k)
            totals["correct"] += float(correct)
            totals["latency_ms"] += latency_ms
            if question.hops >= 2:
                multi_hop["n"] += 1
                multi_hop["correct"] += int(correct)

            per_question.append({
                "qid": question.qid, "hops": question.hops, "recall": round(recall, 3),
                "correct": correct, "answer": answer.text,
                "diagnostics": retrieval.diagnostics,
            })
            logger.info("%s %s: recall=%.2f correct=%s", name, question.qid, recall, correct)

        n = len(eval_questions)
        results[name] = {
            "recall_at_k": round(totals["recall"] / n, 4),
            "hit_rate": round(totals["hit"] / n, 4),
            "mrr": round(totals["mrr"] / n, 4),
            "ndcg_at_k": round(totals["ndcg"] / n, 4),
            "answer_acc": round(totals["correct"] / n, 4),
            "multi_hop_acc": round(multi_hop["correct"] / multi_hop["n"], 4)
                             if multi_hop["n"] else None,
            "avg_latency_ms": round(totals["latency_ms"] / n, 1),
            "n": n,
            "per_question": per_question,
        }

    ranked_methods = sorted(results, key=lambda m: (results[m]["answer_acc"],
                                                    results[m]["recall_at_k"]), reverse=True)
    usage = getattr(runtime.llm, "total_usage", None)
    return {
        "methods": results,
        "ranked": ranked_methods,
        "_meta": {
            "k": k, "n_questions": len(eval_questions), "n_docs": len(runtime.corpus),
            "judge_samples": judge_samples,
            "llm_calls": getattr(runtime.llm, "call_count", None),
            "tokens": {"input": usage.input_tokens, "output": usage.output_tokens}
                      if usage else None,
        },
    }
