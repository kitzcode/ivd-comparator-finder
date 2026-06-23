"""
The Corpus protocol: how a source plugs into the reasoning core.

A corpus owns three things the generic core cannot know:
  - how to gather candidate chunks for a query scope (its storage/index),
  - how its chunks should be scored (RetrievalConfig),
  - how answers should be grounded against it (GroundingContract).

Implementations live under corpora/ (e.g. corpora.fda_510k, corpora.fda_guidance).
The core depends only on this protocol, never on a concrete corpus.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from .models import Chunk
from .retrieve import RetrievalConfig
from .contract import GroundingContract


@runtime_checkable
class Corpus(Protocol):
    name: str

    def candidates(self, scope: Optional[dict[str, Any]] = None) -> list[Chunk]:
        """Return the candidate chunks in scope (unranked). An empty/None scope
        means the whole corpus (may be slow on large corpora)."""
        ...

    @property
    def retrieval_config(self) -> RetrievalConfig:
        ...

    @property
    def grounding(self) -> GroundingContract:
        ...
