"""Agentic RAG — a ReAct-style tool-using retrieval agent (Yao et al. 2023, arXiv:2210.03629).

The LLM holds retrieval tools (dense search, BM25, document reader, corpus catalog) and drives its
own investigation: think, call a tool, read the observation, repeat — chaining its own follow-up
lookups on multi-hop questions. Everything it touches lands in an EvidenceLog, which becomes the
ranked retrieval result.

Package contract: ``Config`` (frozen tunables) + ``Pipeline`` (retrieve / answer).
"""
from .agent import AgentStep, Decision, ReActAgent, Trajectory
from .config import Config
from .evidence import EvidenceLog
from .pipeline import Pipeline
from .tools import Tool, ToolRegistry, build_default_registry

__all__ = ["Config", "Pipeline", "EvidenceLog", "Tool", "ToolRegistry",
           "build_default_registry", "ReActAgent", "Trajectory", "AgentStep", "Decision"]
