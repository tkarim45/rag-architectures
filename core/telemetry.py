"""Lightweight observability: structured logging + span tracing + counters.

Every pipeline stage (`retrieve`, `grade`, `generate`, `traverse`, …) runs inside a `span`. Spans
record wall-clock latency, arbitrary attributes, and nest — a trace of one query through corrective
RAG reads like: `pipeline > retrieve.hybrid > grade(x6) > rewrite > retrieve.fusion > generate`.

No external dependency (OpenTelemetry would be the production choice — this mirrors its API shape
so swapping in an OTLP exporter is mechanical: `Span` ≈ otel span, `Tracer.export()` ≈ processor).
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

logger = logging.getLogger("ragarch")


def configure_logging(level: int = logging.INFO, json_lines: bool = False) -> None:
    """Idempotent root setup for the `ragarch` logger. JSON-lines mode for machine consumers."""
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    if json_lines:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s :: %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


@dataclass
class Span:
    name: str
    span_id: str
    parent_id: str | None
    started_at: float
    ended_at: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    children: list["Span"] = field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at) * 1000.0

    def set(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": None if self.duration_ms is None else round(self.duration_ms, 2),
            "attributes": self.attributes,
            "children": [c.to_dict() for c in self.children],
        }


_current_span: ContextVar[Span | None] = ContextVar("ragarch_current_span", default=None)


class Tracer:
    """Collects span trees per query. One tracer per process is fine — spans nest via contextvars,
    so concurrent pipelines (threads/async) don't corrupt each other's trees."""

    def __init__(self) -> None:
        self.roots: list[Span] = []
        self._counters: dict[str, float] = {}

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        parent = _current_span.get()
        s = Span(name=name, span_id=uuid.uuid4().hex[:12],
                 parent_id=parent.span_id if parent else None,
                 started_at=time.perf_counter(), attributes=dict(attributes))
        if parent is not None:
            parent.children.append(s)
        else:
            self.roots.append(s)
        token = _current_span.set(s)
        try:
            yield s
        finally:
            s.ended_at = time.perf_counter()
            _current_span.reset(token)
            logger.debug("span %s finished in %.1fms", name, s.duration_ms or 0.0,
                         extra={"extra_fields": {"span": name, **s.attributes}})

    def count(self, name: str, value: float = 1.0) -> None:
        self._counters[name] = self._counters.get(name, 0.0) + value

    @property
    def counters(self) -> dict[str, float]:
        return dict(self._counters)

    def reset(self) -> None:
        self.roots.clear()
        self._counters.clear()

    def export(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.roots]


#: Process-wide default tracer. Pipelines accept an injected tracer but fall back to this one so
#: casual usage still gets traces for free.
tracer = Tracer()
