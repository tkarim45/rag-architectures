"""EvidenceLog — the ordered record of everything the agent touched, and how it becomes a ranking.

The ReAct loop's output for the *benchmark* is not its final answer but its evidence trail: which
chunks its searches surfaced and which documents it read in full. Tools write here on every
execution; the pipeline reads back a ranked ``ScoredChunk`` list for the ``RetrievalResult``.

Ranking rationale (frequency + recency blend)
---------------------------------------------
Raw retriever scores are useless across an agent trajectory: dense cosine, BM25 term sums and
whole-document reads are three incomparable scales, produced by *different queries* at different
steps. What the trajectory itself tells us is behavioral, and two behavioral signals correlate
with answer-bearing evidence:

* **touch frequency** — evidence surfaced by several independent probes (different queries, or
  dense *and* keyword search agreeing) is corroborated, exactly the voting intuition behind rank
  fusion (RRF). One touch = one vote.
* **recency** — later steps happen *after* the agent has read intermediate evidence and narrowed
  in; the follow-up ``read_document`` at step 3 of a hop chain is nearly always the answer-bearing
  document, while step-0 search hits are the un-narrowed first guess.

The blend is ``touches + recency_weight * (last_step + 1) / (n_steps + 1)`` with
``recency_weight < 1`` **guaranteed by config validation**, so the recency term is strictly a
tie-breaker: a chunk touched twice always outranks a chunk touched once, and among equally-touched
chunks the one the agent engaged with latest wins. Deterministic final tie-break: first-seen order.

Documents are ranked by the *sum* of their chunks' blend scores (a doc corroborated through many
distinct chunks beats a doc seen through one), and the returned chunk list is grouped by document
rank so ``RetrievalResult.doc_ids`` (first-occurrence order over chunks) reproduces exactly the
documented doc ranking.

Reading a full document records a synthetic whole-document chunk (``chunk_id=f"{doc_id}::doc"``)
so provenance flows through the same ``ScoredChunk`` shape as search hits and the context builder
can hand the generator the complete text the agent actually read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from core import Chunk, Document, ScoredChunk


@dataclass
class _Entry:
    """Mutable per-chunk accumulator. The first-seen chunk object is kept (identical content);
    later touches only update the counters that feed the blend score."""

    scored: ScoredChunk
    tool: str
    first_step: int
    last_step: int
    touches: int
    order: int                          #: first-seen sequence number — the deterministic tie-break


@dataclass
class EvidenceLog:
    """Ordered, deduplicated record of every chunk/document the agent's tools touched.

    The agent loop calls :meth:`begin_step` before executing each action; tools then record
    without needing to know the step index — keeping the tool signatures pure "args in, text out".
    """

    recency_weight: float = 0.9
    _entries: dict[str, _Entry] = field(default_factory=dict)
    _step: int = 0
    _max_step: int = 0

    # ---- write path (agent loop + tools) -------------------------------------------------

    def begin_step(self, step: int) -> None:
        """Set the trajectory step that subsequent recordings are attributed to."""
        self._step = step
        self._max_step = max(self._max_step, step)

    def record_chunks(self, hits: Iterable[ScoredChunk], *, tool: str) -> None:
        """Record search-tool hits at the current step."""
        for hit in hits:
            self._touch(hit, tool)

    def record_document(self, document: Document, *, tool: str) -> None:
        """Record a full-document read as a synthetic whole-doc chunk.

        ``chunk_id=f"{doc_id}::doc"`` follows the core ``"{doc_id}::spec"`` convention, so
        citation resolution (``chunk_id.split("::")[0]``) and doc-level metrics work unchanged.
        """
        chunk = Chunk(chunk_id=f"{document.doc_id}::doc", doc_id=document.doc_id,
                      index_text=document.text, display_text=document.text,
                      metadata={"synthetic": True, "title": document.title})
        self._touch(ScoredChunk(chunk=chunk, score=1.0, retriever=tool), tool)

    def _touch(self, hit: ScoredChunk, tool: str) -> None:
        entry = self._entries.get(hit.chunk_id)
        if entry is None:
            self._entries[hit.chunk_id] = _Entry(scored=hit, tool=tool, first_step=self._step,
                                                 last_step=self._step, touches=1,
                                                 order=len(self._entries))
        else:
            entry.touches += 1
            entry.last_step = max(entry.last_step, self._step)

    # ---- read path (pipeline) ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def _blend(self, entry: _Entry) -> float:
        """Touch frequency dominant, recency as a strict tie-breaker (see module docstring)."""
        recency = (entry.last_step + 1) / (self._max_step + 1)
        return entry.touches + self.recency_weight * recency

    def ranked_doc_ids(self) -> list[str]:
        """Documents ranked by summed chunk blend scores (frequency + recency, see docstring)."""
        scores: dict[str, float] = {}
        first_seen: dict[str, int] = {}
        for entry in self._entries.values():
            doc_id = entry.scored.doc_id
            scores[doc_id] = scores.get(doc_id, 0.0) + self._blend(entry)
            first_seen.setdefault(doc_id, entry.order)
        return sorted(scores, key=lambda d: (-scores[d], first_seen[d]))

    def ranked_chunks(self, limit: int) -> list[ScoredChunk]:
        """Top evidence as ``ScoredChunk``s, re-scored with the blend.

        Grouped by document rank first, then by per-chunk blend within a document, so the derived
        ``RetrievalResult.doc_ids`` equals :meth:`ranked_doc_ids` (truncated to the docs whose
        chunks fit in ``limit``). The original retriever score is discarded on purpose — it is not
        comparable across trajectory steps (module docstring).
        """
        doc_rank = {doc_id: i for i, doc_id in enumerate(self.ranked_doc_ids())}
        ordered = sorted(self._entries.values(),
                         key=lambda e: (doc_rank[e.scored.doc_id], -self._blend(e), e.order))
        return [ScoredChunk(chunk=e.scored.chunk, score=round(self._blend(e), 4),
                            retriever=f"agentic:{e.tool}")
                for e in ordered[:limit]]

    def as_dicts(self) -> list[dict[str, Any]]:
        """Full evidence trail for diagnostics, in first-seen order."""
        return [{"chunk_id": e.scored.chunk_id, "doc_id": e.scored.doc_id, "tool": e.tool,
                 "first_step": e.first_step, "last_step": e.last_step, "touches": e.touches,
                 "blend": round(self._blend(e), 4)}
                for e in sorted(self._entries.values(), key=lambda e: e.order)]
