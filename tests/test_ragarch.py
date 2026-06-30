"""Pure-logic tests — no network, no model download. LLM/embedding calls are monkeypatched so CI
runs fast and offline. The real embedding + Bedrock behaviour is validated by an actual benchmark
run (see the README), not in CI."""
import numpy as np

from common.corpus import docs, questions
from common.evaluate import hit_at_k, recall_at_k
from common.index import CHUNKERS
from common.retrieval import rrf


def test_corpus_gold_docs_all_exist():
    ids = {d.id for d in docs()}
    for q in questions():
        assert q.gold_docs, q.qid
        for g in q.gold_docs:
            assert g in ids, (q.qid, g)
    # at least some genuinely multi-hop questions exist
    assert any(q.hops >= 2 for q in questions())


def test_chunkers_map_to_docs():
    ds = docs()
    ids = {d.id for d in ds}
    for name in ("whole", "sentence", "sentence_window", "parent_child"):
        chunks = CHUNKERS[name](ds)
        assert chunks
        assert all(c.doc_id in ids for c in chunks)
    # parent_child returns the full parent doc text
    pc = CHUNKERS["parent_child"](ds)
    by_doc = {d.id: d.text for d in ds}
    assert all(c.return_text == by_doc[c.doc_id] for c in pc)
    # sentence granularity produces more chunks than whole-doc
    assert len(CHUNKERS["sentence"](ds)) > len(CHUNKERS["whole"](ds))


def test_rrf_rewards_consensus():
    a = ["x", "y", "z"]
    b = ["y", "x", "w"]
    fused = rrf([a, b])
    assert fused[0] in ("x", "y")          # items high in both lists win
    assert set(fused) == {"x", "y", "z", "w"}


def test_recall_and_hit():
    assert recall_at_k(["d1", "d2", "d9"], ["d2", "d3"], k=5) == 0.5
    assert hit_at_k(["d1", "d2"], ["d2"], k=5) is True
    assert hit_at_k(["d1", "d2"], ["d9"], k=5) is False
    assert recall_at_k(["d9", "d8"], ["d1"], k=2) == 0.0


def test_graphrag_build_and_multihop(monkeypatch):
    # fake entity extraction: entities = any corpus proper-noun keywords present in the text
    import graphrag as gr
    KEYS = ["veyra", "quorrel", "talix", "brightfen", "orsa", "mara lindqvist",
            "idris okonkwo", "northwind capital", "pell metrics", "petra voss",
            "cascade", "sun park"]

    def fake_complete(prompt, max_tokens=120):
        text = prompt.lower()
        return ", ".join(k for k in KEYS if k in text)

    import common.providers as prov
    monkeypatch.setattr(prov, "complete", fake_complete)   # graphrag calls common.providers.complete
    graph = gr.build(docs())
    # Talix doc (d3) and Brightfen doc (d4) share the 'brightfen'/'talix' entities → connected
    assert graph.g.has_edge("d3", "d4") or graph.g.has_edge("d4", "d3")

    # multi-hop: "company behind the database Quorrel uses" should surface Brightfen's founder doc
    ranked, ctx = gr.retrieve("Who founded the company behind the Talix database?", graph, k=5)
    assert "d4" in ranked          # Brightfen doc reached via graph traversal
    assert ctx


def test_all_architectures_expose_run():
    import adaptive, agentic, corrective, graphrag, hybrid, hyde
    import multi_query, naive, rag_fusion, raptor, rerank, sparse
    for mod in (naive, sparse, hybrid, rerank, multi_query, rag_fusion, hyde, graphrag, raptor,
                corrective, adaptive, agentic):
        assert callable(mod.run), mod.__name__
