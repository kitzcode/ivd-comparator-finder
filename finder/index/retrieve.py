"""
Retrieval: find the most relevant SummaryChunks for a query.

v2 strategy: section-targeted keyword search (no embeddings).
- Pre-filter: scope to the requested K-number(s) or product code(s).
- Score: count query term hits in each chunk, weight by section relevance.
- Return the top-k chunks, ranked by score.

Embeddings are deferred until keyword retrieval proves insufficient.
"""

from __future__ import annotations

import re
from typing import Optional

from ..models import SummaryChunk
from .store import load_chunks, load_chunks_for_product_code

# Sections that contain performance data get a score boost
_SECTION_BOOST = {
    "Performance Testing": 2.0,
    "Conclusions / Limitations": 1.5,
    "Intended Use / Device Description": 1.2,
    "Substantial Equivalence": 1.1,
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _score_chunk(chunk: SummaryChunk, query_tokens: list[str]) -> float:
    chunk_tokens = _tokenize(chunk.text)
    token_set = set(chunk_tokens)
    hits = sum(1 for t in query_tokens if t in token_set)
    boost = _SECTION_BOOST.get(chunk.section, 1.0)
    return hits * boost


def retrieve(
    query: str,
    k_numbers: Optional[list[str]] = None,
    product_codes: Optional[list[str]] = None,
    top_k: int = 8,
    sections: Optional[list[str]] = None,
) -> list[SummaryChunk]:
    """
    Return the top_k most relevant SummaryChunks for query.

    Scope is determined by k_numbers and/or product_codes (both optional;
    if neither is given, searches all indexed chunks — slow on large corpora).
    Pass sections to restrict to specific section types.
    """
    # Gather candidate chunks
    candidates: list[SummaryChunk] = []
    seen_k: set[str] = set()

    if k_numbers:
        for k in k_numbers:
            if k not in seen_k:
                candidates.extend(load_chunks(k))
                seen_k.add(k)

    if product_codes:
        for pc in product_codes:
            for chunk in load_chunks_for_product_code(pc):
                if chunk.k_number not in seen_k:
                    candidates.append(chunk)
            # Track by product code scope instead of individual K-numbers here
            # (k_numbers set above may not cover all PCs)

    if not k_numbers and not product_codes:
        # Full corpus scan — walk all chunk files
        from pathlib import Path
        from ..index import store as _store
        for k in _store.list_indexed():
            candidates.extend(load_chunks(k))

    # Section filter
    if sections:
        section_set = {s.lower() for s in sections}
        candidates = [c for c in candidates if c.section.lower() in section_set]

    if not candidates:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return candidates[:top_k]

    scored = [(c, _score_chunk(c, query_tokens)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Filter out zero-score chunks unless there's nothing better
    nonzero = [(c, s) for c, s in scored if s > 0]
    result_pairs = nonzero if nonzero else scored
    return [c for c, _ in result_pairs[:top_k]]
