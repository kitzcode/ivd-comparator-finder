"""
FDA 510(k) corpus adapter.

Adapts the finder substrate (openFDA-derived 510(k) Summary chunks, stored and
scored under finder/) to the grounded_rag.Corpus protocol. This is the first of
two corpora that share the same reasoning engine.
"""

from __future__ import annotations

from .corpus import (
    FDA510kCorpus,
    FDA_510K_CONTRACT,
    summary_chunk_to_chunk,
)

__all__ = ["FDA510kCorpus", "FDA_510K_CONTRACT", "summary_chunk_to_chunk"]
