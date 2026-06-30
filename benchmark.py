"""Wire every architecture package into one comparable shape — `run(query, bundle) -> (docs,
context, meta)` — then run them all over the labeled eval set, generate an answer from each method's
context, and score retrieval recall + LLM-judged answer correctness.

One corpus, one eval set, many architectures, honest side-by-side numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import adaptive, agentic, chunking, corrective, graphrag, hybrid, hyde
import multi_query, naive, rag_fusion, raptor, rerank, sparse
from common import generate, providers
from common.corpus import docs as corpus_docs, questions as corpus_questions
from common.evaluate import hit_at_k, recall_at_k
from common.index import CHUNKERS, Index


@dataclass
class Bundle:
    indexes: dict
    graph: object
    raptor: object


# chunker indexes the chunking architecture needs, beyond the default "sentence"
CHUNK_VARIANTS = ["sentence_window", "parent_child", "contextual"]


def build_bundle(with_graph=True, with_raptor=True, with_chunk_variants=True) -> Bundle:
    ds = corpus_docs()
    names = ["sentence"] + (CHUNK_VARIANTS if with_chunk_variants else [])
    indexes = {n: Index(CHUNKERS[n](ds)) for n in names}
    g = graphrag.build(ds) if with_graph else None
    r = raptor.build(ds) if with_raptor else None
    return Bundle(indexes, g, r)


def methods(bundle: Bundle) -> dict:
    reg = {
        "naive": naive.run, "sparse": sparse.run, "hybrid": hybrid.run, "rerank": rerank.run,
        "multi_query": multi_query.run, "rag_fusion": rag_fusion.run, "hyde": hyde.run,
        "corrective": corrective.run,
    }
    for ck in CHUNK_VARIANTS:
        if ck in bundle.indexes:
            reg[f"chunk:{ck}"] = chunking.run_for(ck)
    if bundle.graph is not None:
        reg["graphrag"] = graphrag.run
        reg["adaptive"] = adaptive.run
        reg["agentic"] = agentic.run
    if bundle.raptor is not None:
        reg["raptor"] = raptor.run
    return reg


def run(method_names=None, limit=None, k=5, bundle: Bundle = None) -> dict:
    qs = corpus_questions()
    if limit:
        qs = qs[:limit]
    bundle = bundle or build_bundle()
    reg = methods(bundle)
    names = method_names or list(reg)

    results = {}
    for name in names:
        fn = reg[name]
        rec = hits = corr = 0.0
        per = []
        for q in qs:
            docs, ctx, meta = fn(q.question, bundle)
            r = recall_at_k(docs, q.gold_docs, k)
            h = hit_at_k(docs, q.gold_docs, k)
            ans = generate.answer(q.question, ctx)
            ok = providers.judge_correct(q.question, ans, q.answer)
            rec += r; hits += int(h); corr += int(ok)
            per.append({"qid": q.qid, "recall": round(r, 3), "hit": h, "correct": ok,
                        "hops": q.hops, "answer": ans})
        n = len(qs)
        results[name] = {"recall_at_k": round(rec / n, 4), "hit_rate": round(hits / n, 4),
                         "answer_acc": round(corr / n, 4), "n": n, "per_question": per}
    ranked = sorted(results, key=lambda m: results[m]["answer_acc"], reverse=True)
    return {"methods": results, "ranked": ranked,
            "_meta": {"k": k, "n_questions": len(qs), "n_docs": len(corpus_docs())}}
