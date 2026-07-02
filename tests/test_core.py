"""Core framework tests — fully offline (FakeLLM + HashingEmbedder + NumPy store)."""
from __future__ import annotations

import numpy as np
import pytest

from core import (ContextBuilder, CorrectnessJudge, FakeLLM, Query, RetrievalResult, Runtime,
                  ScoredChunk, StructuredCaller, extract_json, hit_at_k, mrr, ndcg_at_k,
                  recall_at_k, rrf, weighted_fusion)
from core.config import RetryPolicy
from core.embeddings import CachingEmbedder, HashingEmbedder
from core.errors import ProviderError, StructuredOutputError
from core.ingestion import build_chunker
from core.llm.base import BaseLLM, Completion, CompletionRequest
from core.stores import Analyzer, BM25Index, NumpyStore
from core.types import Chunk, Document


@pytest.fixture()
def runtime() -> Runtime:
    return Runtime.for_testing()


@pytest.fixture()
def index(runtime: Runtime):
    return runtime.build_index("sentence")


# ---- chunkers ---------------------------------------------------------------------------

def test_chunkers_produce_expected_shapes(runtime: Runtime):
    docs = runtime.corpus
    sentence = build_chunker("sentence").chunk(docs)
    assert all(c.index_text == c.display_text for c in sentence)

    window = build_chunker("sentence_window", window=1).chunk(docs)
    assert any(len(c.display_text) > len(c.index_text) for c in window)

    parent = build_chunker("parent_child").chunk(docs)
    assert all(c.display_text == runtime.corpus[0].text
               for c in parent if c.doc_id == runtime.corpus[0].doc_id)

    fixed = build_chunker("fixed", max_chars=120, overlap_chars=30).chunk(docs)
    assert all(len(c.index_text) <= 240 for c in fixed)  # windows respect the cap loosely


def test_contextual_chunker_prefixes_llm_context(runtime: Runtime):
    llm = FakeLLM().on("situate", "CTX-MARKER.")
    chunks = build_chunker("contextual", llm=llm).chunk(runtime.corpus[:1])
    assert all(c.index_text.startswith("CTX-MARKER.") for c in chunks)
    assert all(not c.display_text.startswith("CTX-MARKER.") for c in chunks)


def test_unknown_chunker_raises():
    with pytest.raises(KeyError):
        build_chunker("nope")


# ---- stores ------------------------------------------------------------------------------

def test_numpy_store_search_and_filter():
    store = NumpyStore()
    vecs = np.eye(3, dtype=np.float32)
    store.add(["a", "b", "c"], vecs, [{"kind": "x"}, {"kind": "y"}, {"kind": "x"}])
    hits = store.search(vecs[0], 2)
    assert hits[0].id == "a" and hits[0].score == pytest.approx(1.0)
    filtered = store.search(vecs[0], 2, where=lambda m: m.get("kind") == "y")
    assert [h.id for h in filtered] == ["b"]


def test_store_rejects_duplicates_and_empty_search():
    store = NumpyStore()
    store.add(["a"], np.ones((1, 4), dtype=np.float32))
    with pytest.raises(Exception):
        store.add(["a"], np.ones((1, 4), dtype=np.float32))
    empty = NumpyStore()
    with pytest.raises(Exception):
        empty.search(np.ones(4), 1)


def test_bm25_index_ranks_term_matches():
    idx = BM25Index()
    # 3+ docs: with N=2, df=1 Okapi IDF is exactly 0 and every score is 0 — a real BM25 property
    idx.add(["x", "y", "z"],
            ["quorrel stream engine", "unrelated cooking recipe", "gardening tips for spring"])
    hits = idx.search("what is quorrel", 3)
    assert hits and hits[0].id == "x"


def test_analyzer_stems_and_drops_stopwords():
    analyzer = Analyzer()
    assert "the" not in analyzer("the founding of the company")
    assert "found" in analyzer("the founding of the company")


# ---- fusion ------------------------------------------------------------------------------

def _chunk(cid: str, doc: str = "d") -> Chunk:
    return Chunk(cid, doc, cid, cid)


def test_rrf_prefers_consensus():
    a = [ScoredChunk(_chunk("c1"), 0.9), ScoredChunk(_chunk("c2"), 0.5)]
    b = [ScoredChunk(_chunk("c2"), 8.0), ScoredChunk(_chunk("c3"), 4.0)]
    fused = rrf([a, b])
    assert fused[0].chunk_id == "c2"  # appears in both rankings


def test_weighted_fusion_respects_weights():
    a = [ScoredChunk(_chunk("c1"), 1.0), ScoredChunk(_chunk("c2"), 0.0)]
    b = [ScoredChunk(_chunk("c2"), 1.0), ScoredChunk(_chunk("c1"), 0.0)]
    fused = weighted_fusion([a, b], [0.9, 0.1])
    assert fused[0].chunk_id == "c1"


# ---- retrieval result / context ----------------------------------------------------------

def test_doc_ids_dedup_preserves_rank():
    chunks = [ScoredChunk(_chunk("c1", "d1"), 3.0), ScoredChunk(_chunk("c2", "d1"), 2.0),
              ScoredChunk(_chunk("c3", "d2"), 1.0)]
    result = RetrievalResult(Query("q"), chunks)
    assert result.doc_ids == ["d1", "d2"]


def test_context_builder_dedups_and_budgets():
    chunks = [ScoredChunk(Chunk("a", "d1", "x", "same text"), 1.0),
              ScoredChunk(Chunk("b", "d1", "y", "same text"), 0.9),
              ScoredChunk(Chunk("c", "d2", "z", "other"), 0.8)]
    block = ContextBuilder(max_passages=5).build(chunks)
    assert block.text.count("same text") == 1
    assert block.doc_ids == ("d1", "d2")

    tiny = ContextBuilder(max_passages=1).build(chunks)
    assert tiny.truncated


# ---- metrics -----------------------------------------------------------------------------

def test_metrics():
    assert recall_at_k(["a", "b"], ["a", "c"], 2) == 0.5
    assert hit_at_k(["a"], ["a"], 1)
    assert mrr(["x", "a"], ["a"]) == 0.5
    assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == pytest.approx(1.0)
    assert ndcg_at_k(["z", "y"], ["a"], 2) == 0.0


# ---- llm machinery -----------------------------------------------------------------------

def test_fake_llm_rules_and_recording():
    llm = FakeLLM().on("alpha", "A").on("beta", "B")
    assert llm.complete_text("this mentions alpha") == "A"
    assert llm.complete_text("this mentions beta") == "B"
    assert llm.complete_text("nothing") == "FAKE"
    assert llm.call_count == 3


def test_base_llm_retries_then_succeeds():
    class Flaky(BaseLLM):
        def __init__(self):
            super().__init__(RetryPolicy(max_attempts=3, base_delay_s=0.0, max_delay_s=0.0))
            self.attempts = 0

        def _invoke(self, request: CompletionRequest) -> Completion:
            self.attempts += 1
            if self.attempts < 3:
                raise ProviderError("transient", retryable=True)
            return Completion(text="ok")

    flaky = Flaky()
    assert flaky.complete(CompletionRequest(prompt="x")).text == "ok"
    assert flaky.attempts == 3


def test_base_llm_gives_up_on_nonretryable():
    class Dead(BaseLLM):
        def _invoke(self, request):
            raise ProviderError("auth", retryable=False)

    with pytest.raises(ProviderError):
        Dead(RetryPolicy(max_attempts=5, base_delay_s=0.0)).complete(CompletionRequest(prompt="x"))


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('prose then ```json\n{"a": 1}\n``` more') == {"a": 1}
    assert extract_json('noise [1, 2] tail') == [1, 2]


def test_structured_caller_repairs_then_raises():
    llm = FakeLLM(default_response="not json at all")
    with pytest.raises(StructuredOutputError):
        StructuredCaller(llm, max_attempts=2).call("give json")
    assert llm.call_count == 2

    ok = FakeLLM(default_response='{"label": "B"}')
    value = StructuredCaller(ok).call("give json", validator=lambda v: v["label"])
    assert value == "B"


def test_judge_majority_voting():
    llm = FakeLLM(default_response="YES")
    judge = CorrectnessJudge(llm, samples=3)
    assert judge.is_correct("q", "a", "ref")
    assert llm.call_count == 3
    with pytest.raises(ValueError):
        CorrectnessJudge(llm, samples=2)


# ---- embeddings / caching ----------------------------------------------------------------

def test_hashing_embedder_similarity():
    emb = HashingEmbedder()
    vecs = emb.embed_texts(["quorrel stream engine", "quorrel engine", "cooking pasta"])
    sim_close = float(vecs[0] @ vecs[1])
    sim_far = float(vecs[0] @ vecs[2])
    assert sim_close > sim_far


def test_caching_embedder_hits_disk(tmp_path):
    class Counting(HashingEmbedder):
        calls = 0

        def embed_texts(self, texts):
            Counting.calls += len(texts)
            return super().embed_texts(texts)

    cached = CachingEmbedder(Counting(), tmp_path, model_name="test")
    first = cached.embed_texts(["a", "b"])
    again = cached.embed_texts(["a", "b"])
    assert np.allclose(first, again)
    assert Counting.calls == 2  # second call served from disk
    assert cached.hits == 2


# ---- end-to-end through the index ---------------------------------------------------------

def test_index_dense_and_sparse_agree_on_obvious_query(index):
    dense = index.dense_search("Who founded Veyra Systems?", 5)
    sparse = index.sparse_search("Who founded Veyra Systems?", 5)
    assert dense and sparse
    assert {"d1", "d6"} & set(s.doc_id for s in dense[:3] + sparse[:3])
