"""
The Corpus protocol: how a source plugs into the reasoning core.

A corpus owns three things the generic core cannot know:
  - how to gather candidate chunks for a query scope (its storage/index),
  - how its chunks should be scored (RetrievalConfig),
  - how answers should be grounded against it (GroundingContract).

`retrieve(query, filters)` is the brief's one-call retrieval surface; it is the
composition candidates(filters) -> rank(...), provided by RetrieveMixin so every
corpus gets it for free. Implementations live under corpora/ (e.g.
corpora.fda_510k, corpora.fda_guidance). The core depends only on this protocol.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from .models import Chunk
from .retrieve import RetrievalConfig, rank
from .contract import GroundingContract


@runtime_checkable
class Corpus(Protocol):
    name: str

    def candidates(self, scope: Optional[dict[str, Any]] = None) -> list[Chunk]:
        """Return the candidate chunks in scope (unranked). An empty/None scope
        means the whole corpus (may be slow on large corpora)."""
        ...

    def retrieve(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 8,
        sections: Optional[list[str]] = None,
    ) -> list[Chunk]:
        """Gather candidates for `filters` and return the top_k ranked for `query`."""
        ...

    @property
    def retrieval_config(self) -> RetrievalConfig:
        ...

    @property
    def grounding(self) -> GroundingContract:
        ...


class RetrieveMixin:
    """Provides `retrieve()` for any corpus that implements candidates() and
    retrieval_config. Keeps the one-call retrieval surface in one place."""

    def retrieve(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 8,
        sections: Optional[list[str]] = None,
    ) -> list[Chunk]:
        return rank(
            query,
            self.candidates(filters),          # type: ignore[attr-defined]
            self.retrieval_config,             # type: ignore[attr-defined]
            top_k=top_k,
            sections=sections,
        )
