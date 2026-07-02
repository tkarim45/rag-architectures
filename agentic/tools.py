"""The tool layer: what the agent can *do*, described for the LLM and guarded for the loop.

Design rules:

* **Tools never raise into the loop.** ``ToolRegistry.execute`` converts every failure — unknown
  tool, malformed arguments, tool exception — into an ``Error: ...`` observation string. The agent
  reads its own mistake in the scratchpad and can self-correct on the next step; a raised exception
  would kill the whole trajectory over one typo'd tool name.
* **Every execution records its evidence.** Tool closures share one :class:`~agentic.evidence.
  EvidenceLog`; whatever a tool surfaces (search hits, full documents) is logged with the current
  step, which is how the pipeline later reconstructs a ranked retrieval result from the trajectory.
  ``list_documents`` is the deliberate exception — it exposes only the catalog (ids + titles), and
  recording all 14 docs as "touched" would drown the real evidence signal.
* **The registry renders its own prompt block** (:meth:`ToolRegistry.describe_all`), so the prompt
  can never drift from the actual callable surface — add a tool, and the agent learns about it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from core import CorpusIndex, ScoredChunk

from .config import Config
from .evidence import EvidenceLog


@dataclass(frozen=True)
class Tool:
    """One agent-callable capability.

    ``args_schema`` maps argument name → human/LLM-readable description; it is both the prompt
    documentation and the validation contract (all listed arguments are required, no extras).
    ``fn`` takes the schema arguments as keywords and returns the observation text.
    """

    name: str
    description: str
    args_schema: dict[str, str]
    fn: Callable[..., str]


class ToolRegistry:
    """Name → Tool dispatch with prompt rendering and error-as-observation execution."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> "ToolRegistry":
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool
        return self

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def describe_all(self) -> str:
        """Render the tool catalog block for the step prompt."""
        blocks: list[str] = []
        for tool in self._tools.values():
            if tool.args_schema:
                args = ", ".join(f'"{name}": <{desc}>' for name, desc in tool.args_schema.items())
                signature = f"{{{args}}}"
            else:
                signature = "{} (no arguments)"
            blocks.append(f"- {tool.name}: {tool.description}\n  action_input: {signature}")
        return "\n".join(blocks)

    def execute(self, name: str, args: Mapping[str, Any] | Any) -> str:
        """Run a tool; ALL failure modes come back as observation strings, never exceptions."""
        tool = self._tools.get(name)
        if tool is None:
            return (f"Error: unknown tool {name!r}. Available tools: "
                    f"{', '.join(self._tools)}.")
        if not isinstance(args, Mapping):
            return (f"Error: action_input for {name!r} must be a JSON object, got "
                    f"{type(args).__name__}. Expected arguments: {sorted(tool.args_schema)}.")
        missing = sorted(set(tool.args_schema) - set(args))
        unexpected = sorted(set(args) - set(tool.args_schema))
        if missing or unexpected:
            problems = []
            if missing:
                problems.append(f"missing arguments {missing}")
            if unexpected:
                problems.append(f"unexpected arguments {unexpected}")
            return (f"Error: bad arguments for {name!r}: {'; '.join(problems)}. "
                    f"Expected exactly: {sorted(tool.args_schema)}.")
        try:
            return tool.fn(**dict(args))
        except Exception as e:  # noqa: BLE001 — the loop must survive any tool failure
            return f"Error: tool {name!r} failed: {e}"


# ------------------------------------------------------------------------------------------
# Built-in tools over the CorpusIndex
# ------------------------------------------------------------------------------------------

def _snippet(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _render_hits(hits: list[ScoredChunk], snippet_chars: int) -> str:
    if not hits:
        return "No results. Try different terms or another tool."
    return "\n".join(
        f"[chunk {h.chunk_id} | doc {h.doc_id} | score {h.score:.3f}] "
        f"{_snippet(h.chunk.display_text, snippet_chars)}"
        for h in hits)


def build_default_registry(index: CorpusIndex, evidence: EvidenceLog,
                           config: Config) -> ToolRegistry:
    """The standard retrieval toolset: two searches (semantic + lexical), a document reader for
    the follow-up after a hit, and a corpus catalog for orientation. Closures share the injected
    ``EvidenceLog`` so every execution leaves an auditable evidence trail."""

    def search(query: str) -> str:
        hits = index.dense_search(str(query), config.search_k)
        evidence.record_chunks(hits, tool="search")
        return _render_hits(hits, config.snippet_chars)

    def keyword_search(query: str) -> str:
        hits = index.sparse_search(str(query), config.keyword_k)
        evidence.record_chunks(hits, tool="keyword_search")
        return _render_hits(hits, config.snippet_chars)

    def read_document(doc_id: str) -> str:
        doc_id = str(doc_id)
        try:
            document = index.document(doc_id)
        except KeyError:
            return (f"Error: no document with id {doc_id!r}. "
                    "Call list_documents to see the valid ids.")
        evidence.record_document(document, tool="read_document")
        return f"# {document.title} (doc {document.doc_id})\n{document.text}"

    def list_documents() -> str:
        return "\n".join(f"{d.doc_id}: {d.title}" for d in index.documents)

    registry = ToolRegistry()
    registry.register(Tool(
        name="search",
        description=("Dense semantic search over the corpus. Finds passages by meaning, even "
                     "with different wording. Returns top chunks with chunk ids and doc ids."),
        args_schema={"query": "natural-language search phrase"},
        fn=search))
    registry.register(Tool(
        name="keyword_search",
        description=("BM25 keyword search. Best for exact names, product terms and rare tokens "
                     "that semantic search may blur. Returns top chunks with chunk and doc ids."),
        args_schema={"query": "keywords or an exact name/term"},
        fn=keyword_search))
    registry.register(Tool(
        name="read_document",
        description=("Read one document's full text by its doc id — the follow-up move after a "
                     "search hit, to see everything around the matching snippet."),
        args_schema={"doc_id": "document id, e.g. d3"},
        fn=read_document))
    registry.register(Tool(
        name="list_documents",
        description="List every document in the corpus (id and title) to orient yourself.",
        args_schema={},
        fn=list_documents))
    return registry
