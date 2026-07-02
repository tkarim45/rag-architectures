"""The query-complexity classifier — the decision-maker Adaptive-RAG stands or falls on.

Jeong et al. 2024 train a T5-large classifier on silver labels (which strategy actually answered
each training question) to predict, per query, the cheapest sufficient strategy: A (no retrieval),
B (single-step retrieval) or C (multi-step iterative retrieval). This package implements the same
three-way decision as a zero-shot structured LLM call: one cheap completion, validated JSON out.

Two policy decisions live here rather than in the executor, so every consumer sees them applied
uniformly and recorded honestly:

* **A → B coercion.** When ``config.allow_no_retrieval`` is False (the default — see
  ``AdaptiveConfig`` for why parametric answers are wrong *by construction* on this closed
  fictional corpus), a model-emitted ``A`` is coerced to ``B``. The raw label and the coercion
  are both kept in the returned `Classification` so diagnostics never lie about what the
  classifier actually said.
* **Fallback direction.** If the classifier's output is unusable even after
  ``StructuredCaller``'s repair retry, we default to ``C`` — the *most* capable route, not the
  cheapest. Misrouting an easy question to C wastes a few calls; misrouting a multi-hop question
  to B silently caps recall (the failure mode that dominates this architecture). When we cannot
  classify, we buy recall with money.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from core import LLM, StructuredCaller, StructuredOutputError, Tracer
from core import tracer as default_tracer

from .config import AdaptiveConfig
from .prompts import CLASSIFIER_PROMPT

Label = Literal["A", "B", "C"]

_VALID_LABELS: frozenset[str] = frozenset({"A", "B", "C"})

#: Where an unusable classification lands: the most capable route (see module docstring).
FALLBACK_LABEL: Label = "C"


@dataclass(frozen=True)
class Classification:
    """One routing decision, with full provenance for diagnostics.

    ``label`` is the *effective* route the executor will run; ``raw_label`` is what the model
    actually emitted before policy (A→B coercion, fallback) was applied. Keeping both means the
    benchmark can score classifier accuracy separately from routing policy.
    """

    label: Label
    raw_label: str
    reason: str
    coerced: bool = False    # True when A was coerced to B by allow_no_retrieval=False
    fallback: bool = False   # True when structured output failed and FALLBACK_LABEL was used


def _parse_classification(value: Any) -> tuple[str, str]:
    """Validator for ``StructuredCaller``: raises on bad shape, returns ``(label, reason)``."""
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object, got {type(value).__name__}")
    label = str(value["label"]).strip().upper()
    if label not in _VALID_LABELS:
        raise ValueError(f"label must be one of A/B/C, got {label!r}")
    reason = str(value.get("reason", "")).strip()
    return label, reason


class ComplexityClassifier:
    """LLM-backed three-way complexity classifier with policy coercion and a safe fallback."""

    def __init__(self, llm: LLM, config: AdaptiveConfig, tracer: Tracer | None = None) -> None:
        self._caller = StructuredCaller(llm)
        self._config = config
        self._tracer = tracer or default_tracer

    def classify(self, question: str) -> Classification:
        """Label ``question`` A/B/C. Never raises on classifier misbehavior — a routing
        architecture that crashes when its router stumbles is worse than one that routes
        conservatively, so structured-output failure degrades to ``FALLBACK_LABEL``."""
        with self._tracer.span("adaptive.classify") as span:
            try:
                raw_label, reason = self._caller.call(
                    CLASSIFIER_PROMPT.format(question=question),
                    validator=_parse_classification, max_tokens=200)
                fallback = False
            except StructuredOutputError:
                raw_label = FALLBACK_LABEL
                reason = "classifier output unusable after repair retry; defaulted to the most capable route"
                fallback = True
            label: Label = raw_label  # type: ignore[assignment]  # validated against _VALID_LABELS
            coerced = False
            if label == "A" and not self._config.allow_no_retrieval:
                label, coerced = "B", True
            span.set("raw_label", raw_label)
            span.set("label", label)
            span.set("coerced", coerced)
            span.set("fallback", fallback)
        return Classification(label=label, raw_label=raw_label, reason=reason,
                              coerced=coerced, fallback=fallback)
