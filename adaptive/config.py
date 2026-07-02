"""Tunables for Adaptive-RAG (Jeong et al. 2024).

Adaptive-RAG's whole premise is *spend per query, not per system*: a complexity classifier routes
each question to the cheapest strategy predicted to be sufficient. The config therefore splits
cleanly into (1) the routing policy, (2) per-route retrieval budgets, and (3) the shared context
budget every route feeds into. Frozen so one benchmark run's settings are immutable facts — a
routed architecture is hard enough to debug without its knobs moving underneath you.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveConfig:
    """Configuration for the Adaptive-RAG pipeline.

    Attributes:
        allow_no_retrieval: Whether the classifier's label ``A`` (answer from parametric memory,
            no corpus lookup) is allowed to reach the executor. **Defaults to False** because this
            repo's corpus is closed and fictional by construction: no fact about Veyra, Quorrel or
            Brightfen exists in any model's pretraining data, so a parametric answer is *always*
            wrong here — route A can only ever produce hallucination or (with our strictly
            grounded generator) an abstention. The route exists and is fully implemented because
            the architecture is general: over a corpus that overlaps world knowledge (the paper's
            open-domain setting), skipping retrieval for "what is the capital of France?" is the
            single biggest cost win Adaptive-RAG offers. When False, label A is coerced to B and
            the coercion is recorded in diagnostics.
        single_k: Top-k for the ``B`` (single-step) route — one dense pass, exactly the naive
            baseline's shape. This is the budget most queries are expected to ride.
        per_iteration_k: How many fused (dense + BM25, RRF) candidates each multi-step iteration
            considers, *before* dedup against accumulated evidence. Small on purpose: each hop
            should contribute a few precise new facts, not re-flood the evidence pool.
        max_iterations: Hard ceiling on follow-up rounds in the ``C`` (multi-step) route. Each
            iteration costs one LLM decision call plus one fused retrieval; 3 covers every 2–3 hop
            chain in the corpus while bounding worst-case latency and spend.
        rrf_k: The Reciprocal Rank Fusion damping constant used when fusing dense and BM25
            rankings inside the multi-step route. 60 is the canonical value (Cormack et al. 2009);
            lower sharpens the head, higher flattens toward consensus.
        final_k: Cap on evidence chunks the retriever returns (and hence on what retrieval
            metrics score and the context builder sees). Multi-step accumulates evidence across
            hops, so its raw pool can exceed a sensible context; the cap keeps the benchmark's
            per-architecture comparison honest. Evidence keeps accumulation order — seed hits
            first, then each hop's finds — so truncation drops the latest, least-vetted material.
        chunker: Core chunking strategy used when the pipeline must build its own index (the
            benchmark normally injects a shared one). ``"sentence"`` matches the repo baseline.
        context_max_passages: Upper bound on passages stuffed into the generation prompt.
        context_max_chars: Character budget (~tokens × 4) for the assembled context block.
        evidence_max_chars_per_passage: Per-passage truncation when accumulated evidence is shown
            to the follow-up decision LLM. The decision call needs the *gist* of each passage, not
            its full text — this keeps the per-iteration prompt cost flat as evidence grows.
    """

    allow_no_retrieval: bool = False
    single_k: int = 5
    per_iteration_k: int = 5
    max_iterations: int = 3
    rrf_k: int = 60
    final_k: int = 8
    chunker: str = "sentence"
    context_max_passages: int = 6
    context_max_chars: int = 6000
    evidence_max_chars_per_passage: int = 600
