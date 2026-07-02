from typing import Protocol, runtime_checkable

from ..types import Query, RetrievalResult
from .context import ContextBuilder
from .fusion import rrf, weighted_fusion


@runtime_checkable
class Retriever(Protocol):
    """The one interface every architecture's retrieval side exposes."""

    def retrieve(self, query: Query) -> RetrievalResult: ...


__all__ = ["Retriever", "ContextBuilder", "rrf", "weighted_fusion"]
