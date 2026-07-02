"""Per-document entity/relation extraction — the offline LLM stage that everything else rests on.

This is the single most consequential call in GraphRAG: an entity the model misses here simply
does not exist downstream, and every traversal path through it is silently gone (the failure mode
behind the corpus's multi-hop misses). The stage therefore goes through `StructuredCaller` — parse,
validate, repair-retry — rather than trusting a free-form reply, and the validator enforces the
graph's invariants at the boundary:

  * entity names are normalized (casefold + whitespace collapse) so "Veyra Systems" and
    "veyra systems" merge into one node across documents;
  * entity types are coerced into the closed vocabulary from `prompts.ENTITY_TYPES`;
  * relations whose endpoints were not extracted as entities are dropped — a dangling edge would
    corrupt the graph, and a hallucinated endpoint is worse than a missing edge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core import Document, LLM, StructuredCaller, Tracer

from .config import Config
from .prompts import ENTITY_TYPES, EXTRACTION_PROMPT


def normalize_entity_name(name: str) -> str:
    """Canonical node key: casefolded, whitespace-collapsed. This is the merge key that unifies
    the same entity across documents (and across the LLM's inconsistent casing)."""
    return " ".join(name.casefold().split())


@dataclass(frozen=True)
class ExtractedEntity:
    """One typed entity as extracted from one document."""

    name: str          # normalized (the graph node key)
    display_name: str  # original surface form, kept for human-readable diagnostics
    type: str          # one of prompts.ENTITY_TYPES
    description: str


@dataclass(frozen=True)
class ExtractedRelation:
    """One directed, typed relation between two extracted entities (endpoints normalized)."""

    source: str
    target: str
    type: str          # short verb phrase ("founded", "acquired", ...)
    description: str


@dataclass(frozen=True)
class DocumentExtraction:
    """Everything the LLM extracted from one document; `doc_id` is the provenance every node and
    edge built from this extraction will carry."""

    doc_id: str
    entities: tuple[ExtractedEntity, ...]
    relations: tuple[ExtractedRelation, ...]


def validate_extraction(payload: Any) -> tuple[tuple[ExtractedEntity, ...],
                                               tuple[ExtractedRelation, ...]]:
    """Validator for `StructuredCaller`: raises ValueError/KeyError/TypeError on bad shape (which
    triggers the repair-retry loop) and returns typed, normalized, invariant-safe values."""
    if not isinstance(payload, dict):
        raise TypeError(f"extraction must be a JSON object, got {type(payload).__name__}")

    entities: list[ExtractedEntity] = []
    seen: set[str] = set()
    for raw in payload.get("entities", []):
        if not isinstance(raw, dict):
            raise TypeError("each entity must be a JSON object")
        display = str(raw["name"]).strip()
        name = normalize_entity_name(display)
        if not name or name in seen:
            continue                                  # empty or duplicate within this doc
        seen.add(name)
        etype = str(raw.get("type", "other")).strip().casefold()
        if etype not in ENTITY_TYPES:
            etype = "other"                           # coerce, don't fail: type is advisory
        entities.append(ExtractedEntity(name=name, display_name=display, type=etype,
                                        description=str(raw.get("description", "")).strip()))

    relations: list[ExtractedRelation] = []
    for raw in payload.get("relations", []):
        if not isinstance(raw, dict):
            raise TypeError("each relation must be a JSON object")
        source = normalize_entity_name(str(raw["source"]))
        target = normalize_entity_name(str(raw["target"]))
        if source not in seen or target not in seen or source == target:
            continue                                  # drop dangling / self edges silently
        rel_type = " ".join(str(raw.get("type", "related_to")).strip().split()) or "related_to"
        relations.append(ExtractedRelation(source=source, target=target, type=rel_type,
                                           description=str(raw.get("description", "")).strip()))

    return tuple(entities), tuple(relations)


@dataclass
class EntityRelationExtractor:
    """Runs the extraction prompt over documents, one structured call per document."""

    llm: LLM
    config: Config
    tracer: Tracer

    def extract(self, document: Document) -> DocumentExtraction:
        with self.tracer.span("graphrag.extract", doc_id=document.doc_id) as span:
            caller = StructuredCaller(self.llm)
            entities, relations = caller.call(
                EXTRACTION_PROMPT.format(doc_id=document.doc_id, title=document.title,
                                         text=document.text),
                validator=validate_extraction,
                max_tokens=self.config.extraction_max_tokens)
            span.set("entities", len(entities))
            span.set("relations", len(relations))
        return DocumentExtraction(doc_id=document.doc_id, entities=entities, relations=relations)
