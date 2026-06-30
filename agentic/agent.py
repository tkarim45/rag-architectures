"""The ReAct loop — the online heart of Agentic RAG.

Each step the LLM reads the question plus the running notes and emits one tool call as JSON. We run
that tool, append its result to the notes, and loop until the LLM says `finish` or we hit the step
cap. The accumulated doc ids (deduped) and notes are what the generator ultimately answers from.

This is the expensive, flexible path: several LLM round-trips per question, with the freedom to chain
its own multi-hop lookups — and the matching risk of looping, dead ends, or stopping too early.
"""
from __future__ import annotations

import json
import re

from common import providers

from .prompts import ACTION_PROMPT
from .tools import graph_search, vector_search

_JSON = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_action(raw: str) -> dict:
    """Pull the first JSON object out of the LLM reply. On any failure, treat it as `finish` so a
    malformed step ends the loop instead of crashing it."""
    m = _JSON.search(raw)
    if not m:
        return {"tool": "finish"}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {"tool": "finish"}
    except (json.JSONDecodeError, ValueError):
        return {"tool": "finish"}


def run_agent(query, index, graph, max_steps: int = 3, tool_k: int = 4) -> tuple[list[str], str, int]:
    """Drive the tool loop. Returns (used_docs[:6], joined notes, steps taken)."""
    notes: list[str] = []
    used_docs: list[str] = []
    steps = 0

    for _ in range(max_steps):
        steps += 1
        action = _parse_action(
            providers.complete(
                ACTION_PROMPT.format(query=query, notes="\n".join(notes) or "(none)"),
                max_tokens=80,
            )
        )
        tool = action.get("tool", "finish")
        if tool == "finish":
            break

        q = action.get("q", query)
        if tool == "graph_search":
            docs, ctx = graph_search(q, graph, tool_k)
        else:                                    # default to vector_search for any other tool name
            tool = "vector_search"
            docs, ctx = vector_search(q, index, tool_k)

        for d in docs:                           # accumulate doc ids, dedup, order-preserving
            if d not in used_docs:
                used_docs.append(d)
        notes.append(f"[{tool} '{q}'] -> {ctx}")

    return used_docs[:6], "\n\n".join(notes), steps
