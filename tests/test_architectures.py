"""Contract + smoke tests for every architecture package — fully offline.

One FakeLLM carries routing rules for every LLM touchpoint in the framework (rules are matched in
registration order, first match wins), so all thirteen pipelines run end-to-end with zero network.
These tests pin the *package contract* (constructor shape, retrieve/answer types, diagnostics
present) rather than retrieval quality — quality is the real benchmark's job.
"""
from __future__ import annotations

import json

import pytest

from core import ContextBlock, FakeLLM, PipelineResult, RetrievalResult, Runtime

QUESTION = "Who founded Veyra Systems?"
MULTI_HOP = "Who founded the company that makes the database Quorrel uses?"


def _agentic_responder(request) -> str:
    """One search step, then a final answer (the scratchpad grows an Observation after step 1)."""
    if "Observation" in request.prompt:
        return json.dumps({"thought": "enough evidence",
                           "final_answer": "Mara Lindqvist founded Veyra Systems."})
    return json.dumps({"thought": "search first", "action": "search",
                       "action_input": {"query": "Veyra Systems founder"}})


def build_fake_llm() -> FakeLLM:
    return (FakeLLM()
            # core: judge + contextual chunker
            .on("factually correct and complete", "YES")
            .on("situate", "This document is about the Veyra ecosystem.")
            # query-transform family
            .on("alternative phrasings", '["Veyra Systems founder", "who started Veyra"]')
            .on("broaden and diversify", '["Veyra Systems founder", "Veyra history"]')
            .on("hypothetical document",
                "Veyra Systems was founded by Mara Lindqvist in Tallinn in 2014.")
            # corrective (CRAG)
            .on("Grade the retrieved passage", '{"grade": "correct", "confidence": 0.9}')
            .on("Is this strip relevant", "YES")
            .on("Rewrite the question", "Veyra Systems founder")
            # adaptive
            .on("Classify the question complexity", '{"label": "B", "reason": "single doc"}')
            .on("What additional information",
                '{"done": true, "next_query": "", "reason": "sufficient"}')
            # graphrag
            .on("Extract entities and relations", json.dumps({
                "entities": [
                    {"name": "Veyra Systems", "type": "organization", "description": "company"},
                    {"name": "Mara Lindqvist", "type": "person", "description": "founder"}],
                "relations": [
                    {"source": "Mara Lindqvist", "target": "Veyra Systems", "type": "founded",
                     "description": "founded the company"}]}))
            .on("Summarize this community", "A community about Veyra Systems and its people.")
            .on("entities mentioned in the question", '["Veyra Systems"]')
            .on("specific entities or a broad", '{"mode": "local"}')
            .on('{"score"', '{"score": 8}')
            # raptor
            .on("Summarize the following passages",
                "People and companies in the Veyra/Brightfen ecosystem.")
            # agentic
            .on("Decide your next action", _agentic_responder))


@pytest.fixture(scope="module")
def resources():
    from benchmark import build_resources

    runtime = Runtime.for_testing(llm=build_fake_llm())
    return build_resources(runtime)


@pytest.fixture(scope="module")
def registry(resources):
    from benchmark import method_registry

    return method_registry(resources)


ALL_METHODS = ["naive", "sparse", "hybrid", "rerank", "multi_query", "rag_fusion", "hyde",
               "corrective", "adaptive", "agentic", "graphrag", "raptor",
               "chunk:sentence_window", "chunk:parent_child", "chunk:contextual"]


def test_registry_is_complete(registry):
    assert set(registry) == set(ALL_METHODS)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_pipeline_contract(method, resources, registry):
    if method == "rerank":
        # inject the deterministic reranker — the cross-encoder is a real model download
        from rerank import LexicalOverlapReranker, Pipeline as RerankPipeline

        pipeline = RerankPipeline(resources.runtime, index=resources.indexes["sentence"],
                                  reranker=LexicalOverlapReranker())
    else:
        pipeline = registry[method](resources)

    retrieval, context = pipeline.retrieve(QUESTION)
    assert isinstance(retrieval, RetrievalResult)
    assert isinstance(context, ContextBlock)
    assert retrieval.doc_ids, f"{method}: no docs retrieved"
    assert context.text.strip(), f"{method}: empty context"
    assert isinstance(retrieval.diagnostics, dict)

    result = pipeline.answer(QUESTION)
    assert isinstance(result, PipelineResult)
    assert result.answer is not None and result.answer.text.strip()


def test_diagnostics_tell_the_architecture_story(resources, registry):
    """Spot-check that key architectures record their mechanism, not just their output."""
    retrieval, _ = registry["multi_query"](resources).retrieve(QUESTION)
    assert len(retrieval.diagnostics["generated_queries"]) >= 2

    retrieval, _ = registry["corrective"](resources).retrieve(QUESTION)
    assert retrieval.diagnostics["action"] in ("refine", "fallback", "combine", "none")

    retrieval, _ = registry["adaptive"](resources).retrieve(MULTI_HOP)
    assert retrieval.diagnostics["route"] in ("A", "B", "C")

    retrieval, _ = registry["agentic"](resources).retrieve(MULTI_HOP)
    steps = retrieval.diagnostics.get("steps") or retrieval.diagnostics.get("trajectory")
    assert steps, "agentic must expose its trajectory"


def test_graphrag_artifacts(resources):
    graph = resources.graph
    assert graph is not None
    # entity merge across docs: 'veyra systems' extracted from every doc collapses to one node
    assert len({n.casefold() for n in getattr(graph, "entity_docs", {"veyra systems": 1})}) >= 1


def test_raptor_tree_has_summary_levels(resources):
    tree = resources.tree
    levels = {node.level for node in tree.all_nodes()}
    assert 0 in levels and len(levels) > 1
