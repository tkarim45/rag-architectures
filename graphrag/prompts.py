"""Every LLM touchpoint in GraphRAG, in one place.

Keeping prompts out of the logic modules makes them auditable and testable: the offline test suite
routes a FakeLLM on the distinctive marker phrase of each prompt, so each phrase below is part of
the package's contract — change a marker and the tests (and any cached completions) must change
with it.

Markers:
    extraction        -> "Extract entities and relations"
    community summary -> "Summarize this community"
    query entities    -> "entities mentioned in the question"
    community rating  -> "Rate how relevant"
    router            -> "specific entities or a broad"
"""
from __future__ import annotations

#: Allowed entity types, shared by the prompt and the extraction validator.
ENTITY_TYPES: tuple[str, ...] = (
    "person", "organization", "product", "technology", "location", "other")

EXTRACTION_PROMPT = """\
Extract entities and relations from the document below.

Return ONLY a JSON object with this exact shape:
{{
  "entities": [
    {{"name": "<entity name>", "type": "<one of: person, organization, product, technology, location, other>", "description": "<one sentence about this entity, grounded in the document>"}}
  ],
  "relations": [
    {{"source": "<entity name>", "target": "<entity name>", "type": "<short verb phrase, e.g. founded, acquired, built_by>", "description": "<one sentence describing the relationship>"}}
  ]
}}

Rules:
- Every relation's source and target MUST appear in the entities list.
- Use the entity's canonical surface form from the document (no pronouns).
- Only extract what the document states; do not infer facts from outside knowledge.

Document (id: {doc_id}, title: {title}):
{text}
"""

COMMUNITY_SUMMARY_PROMPT = """\
Summarize this community of related entities from a knowledge graph.

Write a short, information-dense report (2-4 sentences) covering: what the community is about,
the key entities and their types, and the most important relationships between them. Someone
should be able to answer broad questions about this cluster from your summary alone.

Entities:
{entities}

Relations:
{relations}
"""

QUERY_ENTITY_PROMPT = """\
List the entities mentioned in the question below.

Return ONLY a JSON array of entity name strings, e.g. ["Acme Corp", "Jane Doe"].
Include only concrete named things (people, organizations, products, technologies, locations).
Return [] if the question names no specific entity.

Question: {question}
"""

COMMUNITY_RATING_PROMPT = """\
Rate how relevant this community summary is for answering the question.

Return ONLY a JSON object: {{"score": <integer 0-10>}} where 0 means irrelevant and 10 means the
summary directly contains the information needed.

Question: {question}

Community summary:
{summary}
"""

ROUTER_PROMPT = """\
Decide whether this question asks about specific entities or a broad, corpus-level theme.

- "local": the question centers on one or a few named entities and their direct relationships.
- "global": the question asks for an aggregate, comparison, or theme across many documents.

Return ONLY a JSON object: {{"mode": "local"}} or {{"mode": "global"}}.

Question: {question}
"""
