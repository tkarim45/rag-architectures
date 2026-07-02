"""Tunables for BM25 lexical retrieval.

Unlike the dense baseline, sparse retrieval exposes *analysis* knobs (stemming, stopwords, token
length) alongside the scoring knobs (k1, b) — because BM25 quality lives mostly in the analyzer,
not the formula. The scoring/analysis defaults here deliberately mirror
``core.stores.lexical.BM25Index`` / ``Analyzer`` so that an untouched config can reuse the shared
index's prebuilt BM25 half instead of rebuilding one (see ``matches_core_defaults``).
"""
from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class SparseConfig:
    """Configuration for the BM25 pipeline.

    Scoring (Okapi BM25, Robertson & Zaragoza 2009):
        k1: Term-frequency saturation. Low k1 → a term counts once no matter how often it
            repeats; high k1 → repeated mentions keep adding score. 1.5 is the classic default.
        b: Length normalization. b=1 fully penalizes long chunks, b=0 not at all. 0.75 is the
            near-universal default; lower it if your chunker already produces uniform lengths.

    Analysis (where BM25 quality actually lives):
        stem: Light suffix stripping so "founded" matches "founder"/"founding". Turning it off
            makes matching stricter — better for IDs and codes, worse for prose.
        min_token_len: Drop tokens shorter than this. 2 kills single-letter noise; raise with
            care (it would also kill meaningful short tokens like "ai" at 3).
        extra_stopwords: Domain-specific words to ignore *in addition to* the core default list.
            The highest-leverage knob here: one corpus-ubiquitous term ("company", "system")
            polluting queries can dominate every BM25 score.

    Retrieval / context:
        chunker: Core chunking strategy for lazily-built indexes; ``"sentence"`` matches the
            repo-wide baseline so sparse-vs-dense deltas isolate the retrieval method.
        top_k: Hits returned. BM25 returns only positive-scoring chunks, so fewer than k
            results is a signal (no lexical overlap), not a bug.
        context_max_passages / context_max_chars: Same generator-facing budget as naive, kept
            identical so benchmark deltas are retrieval deltas.
    """

    # scoring
    k1: float = 1.5
    b: float = 0.75
    # analysis
    stem: bool = True
    min_token_len: int = 2
    extra_stopwords: tuple[str, ...] = ()
    # retrieval / context
    chunker: str = "sentence"
    top_k: int = 5
    context_max_passages: int = 5
    context_max_chars: int = 6000

    #: Fields whose defaults mirror the core BM25Index/Analyzer defaults. If any of these is
    #: changed, the shared index's prebuilt BM25 no longer reflects this config and the retriever
    #: must build its own (silently ignoring the config would be the worse bug).
    _BM25_FIELDS = ("k1", "b", "stem", "min_token_len", "extra_stopwords")

    def matches_core_defaults(self) -> bool:
        """True when the shared ``CorpusIndex.bm25`` (built with core defaults) already implements
        this config exactly, so reusing it is correct — not just convenient."""
        defaults = {f.name: f.default for f in fields(self)}
        return all(getattr(self, name) == defaults[name] for name in self._BM25_FIELDS)
