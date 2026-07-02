"""Every LLM touchpoint in the RAPTOR package.

There is exactly one: the cluster summarization prompt used while building the tree. Keeping it
here (per the core quality bar) makes the package's entire LLM surface auditable at a glance and
gives offline tests a stable substring to route on (`FakeLLM().on("Summarize the following
passages", ...)`).
"""
from __future__ import annotations

SUMMARIZE_SYSTEM: str = (
    "You are a careful technical summarizer. You compress passages without inventing facts: "
    "every statement in your summary must be directly supported by the input passages."
)

SUMMARIZE_TEMPLATE: str = """Summarize the following passages into a single coherent paragraph.

Preserve every concrete fact — names of people, companies, products, dates, and the relationships
between them — because this summary will be searched instead of the passages themselves. Do not
add information that is not in the passages. Do not editorialize.

Passages:
{passages}

Summary:"""


def summarize_prompt(texts: list[str]) -> str:
    """Render the cluster-summary prompt over the member node texts, numbered for clarity."""
    passages = "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(texts))
    return SUMMARIZE_TEMPLATE.format(passages=passages)
