"""Prompts for Agentic RAG. The LLM is the controller: at each step it sees the question plus the
notes gathered so far and emits a single tool call as JSON. Keeping the action space tiny and the
output format rigid (JSON only) is what makes the ReAct loop parseable and cheap to run."""
from __future__ import annotations

ACTION_PROMPT = (
    "You are a retrieval agent answering a question by gathering evidence with tools. You have:\n"
    '  {{"tool": "vector_search", "q": "..."}}  - semantic search over the document chunks; '
    "best for finding facts by topic or wording.\n"
    '  {{"tool": "graph_search", "q": "..."}}   - traversal over an entity graph; '
    "best for multi-hop questions that connect entities across documents.\n"
    '  {{"tool": "finish"}}                      - stop searching; you have enough to answer.\n\n'
    "Use vector_search for direct lookups and graph_search when the answer needs you to hop between "
    "related entities. Issue finish as soon as the notes cover the question - do not search "
    "needlessly.\n\n"
    "Question: {query}\n"
    "Notes so far:\n{notes}\n"
    "Choose the single next action. Reply with ONLY the JSON."
)
