"""Answer generation — grounded strictly in the retrieved context (so a wrong answer is a
retrieval failure, not the model free-styling from memory)."""
from __future__ import annotations

from . import providers


def answer(question: str, context: str) -> str:
    if not context.strip():
        return "I don't know."
    return providers.complete(
        "Answer the question using ONLY the context below. If the answer isn't in the context, say "
        "\"I don't know.\" Be concise.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:", max_tokens=200)
