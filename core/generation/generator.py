"""Grounded answer generation.

The generator is deliberately strict: answer ONLY from the supplied context, abstain otherwise.
That policy is what makes the benchmark meaningful — a wrong answer is a *retrieval* failure, not
the model free-styling from parametric memory. Citations are parsed from the model's `[n]` markers
back into doc ids via the ContextBlock provenance.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..llm.base import LLM, CompletionRequest
from ..telemetry import Tracer, tracer as default_tracer
from ..types import ContextBlock, GeneratedAnswer

DEFAULT_SYSTEM = (
    "You are a retrieval-grounded assistant. Answer strictly from the provided context passages. "
    "If the context does not contain the answer, reply exactly: I don't know. Cite the passages "
    "you used with their [n] markers. Be concise.")

ANSWER_TEMPLATE = "Context passages:\n{context}\n\nQuestion: {question}\nAnswer:"

_ABSTAIN = re.compile(r"\bi don'?t know\b", re.IGNORECASE)
_CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class AnswerGenerator:
    llm: LLM
    system: str = DEFAULT_SYSTEM
    max_tokens: int = 300
    tracer: Tracer = field(default_factory=lambda: default_tracer)

    def generate(self, question: str, context: ContextBlock) -> GeneratedAnswer:
        if not context.text.strip():
            return GeneratedAnswer(text="I don't know.", abstained=True)
        with self.tracer.span("generate", passages=len(context.chunk_ids)) as span:
            completion = self.llm.complete(CompletionRequest(
                prompt=ANSWER_TEMPLATE.format(context=context.text, question=question),
                system=self.system, max_tokens=self.max_tokens))
            text = completion.text.strip()
            citations = self._resolve_citations(text, context)
            span.set("abstained", bool(_ABSTAIN.search(text)))
        return GeneratedAnswer(text=text, citations=citations, usage=completion.usage,
                               abstained=bool(_ABSTAIN.search(text)))

    @staticmethod
    def _resolve_citations(text: str, context: ContextBlock) -> tuple[str, ...]:
        """Map `[n]` markers in the answer back to doc ids through the context's chunk order."""
        doc_ids: list[str] = []
        for marker in _CITATION.findall(text):
            position = int(marker) - 1
            if 0 <= position < len(context.chunk_ids):
                # chunk order in the block == passage numbering
                chunk_id = context.chunk_ids[position]
                doc_id = chunk_id.split("::")[0]
                if doc_id not in doc_ids:
                    doc_ids.append(doc_id)
        return tuple(doc_ids)
