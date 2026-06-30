"""Configuration for Agentic RAG. Tunables live here so the pipeline reads like prod code: change
behavior by editing config, not the logic."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    chunker: str = "sentence"   # which offline index (built in common/index.py) the agent searches
    max_steps: int = 3          # cap on agent reasoning steps (each is one LLM round-trip)
    tool_k: int = 4             # results each tool call pulls back
