"""LLM-as-judge for answer correctness.

Judges facts against a reference answer, ignoring phrasing. Supports multi-sample majority voting:
a single judge call flips on borderline answers, and with a 12-question eval set one flip moves a
method ~8 points — voting is the cheap way to stabilize the benchmark's headline numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..llm.base import LLM, CompletionRequest
from ..telemetry import Tracer, tracer as default_tracer

JUDGE_PROMPT = (
    "Question: {question}\n"
    "Reference answer: {reference}\n"
    "Candidate answer: {candidate}\n\n"
    "Is the candidate answer factually correct and complete relative to the reference? Ignore "
    "wording differences; judge only the facts. Reply with exactly YES or NO.")


@dataclass
class CorrectnessJudge:
    llm: LLM
    samples: int = 1              # odd; >1 enables majority voting
    tracer: Tracer = None         # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tracer is None:
            self.tracer = default_tracer
        if self.samples % 2 == 0:
            raise ValueError("samples must be odd for majority voting")

    def is_correct(self, question: str, candidate: str, reference: str) -> bool:
        votes = 0
        with self.tracer.span("judge", samples=self.samples) as span:
            for _ in range(self.samples):
                reply = self.llm.complete(CompletionRequest(
                    prompt=JUDGE_PROMPT.format(question=question, reference=reference,
                                               candidate=candidate),
                    max_tokens=5)).text
                votes += int(reply.strip().upper().startswith("YES"))
            verdict = votes > self.samples // 2
            span.set("votes_yes", votes)
            span.set("verdict", verdict)
        return verdict
