"""Design profiles for every core chunking strategy.

This module is the intellectual content of the package: a structured account of *what each
strategy actually decides*. Every chunker fixes two independent texts per chunk — the
**index text** (what gets embedded and BM25-indexed → match precision) and the **display text**
(what the generator reads on a hit → answer context). Naive chunking couples the two; the
interesting strategies drive them apart, matching on something small and precise while returning
something larger and richer.

All six core strategies are profiled, including the three the benchmark does not run
(`whole`, `sentence`, `fixed`) — they are the coupled baselines the decoupled strategies are
reactions to, and the profiles only make sense side by side. `STRATEGIES` names the three the
benchmark actually builds indexes for.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from core import CHUNKER_REGISTRY

#: The strategies this package benchmarks head-to-head — the three that decouple index text from
#: display text. The exact names are core `CHUNKER_REGISTRY` keys; the benchmark iterates this.
STRATEGIES: tuple[str, ...] = ("sentence_window", "parent_child", "contextual")


@dataclass(frozen=True)
class StrategyProfile:
    """One chunking strategy's design card: what it indexes, what it returns, and the
    precision-vs-context trade it makes. `benchmarked` marks membership in :data:`STRATEGIES`."""

    name: str
    indexed: str
    """What the embedder/BM25 sees — determines match precision."""
    returned: str
    """What the generator reads on a hit — determines answer context."""
    tradeoff: str
    """The precision-vs-context position this strategy stakes out."""
    wins_when: str
    loses_when: str
    benchmarked: bool = False


_PROFILES: tuple[StrategyProfile, ...] = (
    StrategyProfile(
        name="whole",
        indexed="the entire document as one vector",
        returned="the entire document",
        tradeoff="Zero precision, maximal context. One vector must average every topic the "
                 "document touches, so the query signal is diluted by everything else in it.",
        wins_when="documents are short and single-topic — any hit is the whole right answer, "
                  "and there is nothing to fragment.",
        loses_when="documents are long or multi-topic: the averaged embedding matches nothing "
                   "sharply, and a hit floods the context budget with mostly-irrelevant text.",
    ),
    StrategyProfile(
        name="sentence",
        indexed="one sentence",
        returned="the same sentence",
        tradeoff="Maximal precision, starved context. The embedding is exactly the claim being "
                 "matched — and exactly all the generator gets.",
        wins_when="the answer is literally a single sentence (dates, names, definitions) and "
                  "precision@1 is everything.",
        loses_when="the answer needs surrounding text — pronoun antecedents, multi-sentence "
                   "arguments, tables split across sentences. The generator sees a fragment "
                   "and abstains or guesses.",
    ),
    StrategyProfile(
        name="fixed",
        indexed="a ~800-char sentence-aligned window with overlap",
        returned="the same window",
        tradeoff="The industry-default compromise — but it still couples matching and reading "
                 "granularity, so one window size must serve two masters.",
        wins_when="long unstructured prose with no better signal; a solid general-purpose "
                  "baseline when you cannot afford strategy tuning.",
        loses_when="the key sentence shares its window with unrelated text: the embedding is a "
                   "topic average again, and boundary placement decides whether evidence is "
                   "split across two windows.",
    ),
    StrategyProfile(
        name="sentence_window",
        indexed="one sentence",
        returned="the sentence plus ±N neighbor sentences",
        tradeoff="Decouples the seam: sentence-level match precision with paragraph-level "
                 "reading context, at a modest token overhead per hit.",
        wins_when="answers span a few adjacent sentences — the matched claim plus its "
                  "immediate qualifiers, causes, or numbers.",
        loses_when="the supporting context lies outside the window (elsewhere in the document "
                   "or in another document); widening N converges on parent_child costs "
                   "without its guarantees.",
        benchmarked=True,
    ),
    StrategyProfile(
        name="parent_child",
        indexed="one sentence (the child)",
        returned="the entire parent document",
        tradeoff="Small-to-big taken to its limit: sentence precision at match time, whole-doc "
                 "context at read time. Many child hits from one document collapse (via "
                 "display-text dedup) into a single passage.",
        wins_when="answers require synthesizing a whole document — summaries, multi-fact "
                  "questions whose evidence is scattered through one doc.",
        loses_when="context budgets are tight: each hit costs a full document of tokens, so "
                   "two long parents can evict every other source from the prompt.",
        benchmarked=True,
    ),
    StrategyProfile(
        name="contextual",
        indexed="an LLM-written one-line document context prepended to each sentence",
        returned="the bare sentence",
        tradeoff="Anthropic-style contextual retrieval: spend one LLM call per document at "
                 "build time to make isolated chunks self-describing ('the company' → *which* "
                 "company), sharpening both dense and BM25 matching.",
        wins_when="the corpus holds many similar entities and bare sentences are ambiguous "
                  "across documents — the prefix is the disambiguator.",
        loses_when="the LLM prefix is wrong or generic: the same misleading context is stamped "
                   "onto every chunk of the document, poisoning them all at once; build cost "
                   "also scales linearly with corpus size.",
        benchmarked=True,
    ),
)

#: Registry of design profiles, keyed by core chunker name. Read-only by construction.
STRATEGY_PROFILES: Mapping[str, StrategyProfile] = MappingProxyType(
    {p.name: p for p in _PROFILES})


def profile(name: str) -> StrategyProfile:
    """Look up a strategy's design profile, with the same error contract as core's registry."""
    try:
        return STRATEGY_PROFILES[name]
    except KeyError:
        raise KeyError(
            f"unknown strategy {name!r}; known: {sorted(STRATEGY_PROFILES)}") from None


# Keep this module honest: if core grows a chunking strategy we have not profiled (or drops one
# we still document), fail loudly at import time instead of silently drifting out of sync.
_DRIFT = set(CHUNKER_REGISTRY).symmetric_difference(STRATEGY_PROFILES)
if _DRIFT:
    raise RuntimeError(
        f"strategy profiles out of sync with core CHUNKER_REGISTRY: {sorted(_DRIFT)}")
if not set(STRATEGIES) <= set(CHUNKER_REGISTRY):
    raise RuntimeError(
        f"STRATEGIES contains names missing from core CHUNKER_REGISTRY: "
        f"{sorted(set(STRATEGIES) - set(CHUNKER_REGISTRY))}")
