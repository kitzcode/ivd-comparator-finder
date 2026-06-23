"""
FDA 510(k) retrieval adapter.

The scoring engine now lives in the corpus-agnostic core (grounded_rag.retrieve).
This module keeps the FDA-specific pieces:
  - scope gathering (load chunks by K-number / product code, or full corpus),
  - the FDA RetrievalConfig (performance-section boosts, IVD domain-term weights,
    and IVD stopwords like "device"/"510"/"k").

It returns SummaryChunk objects unchanged, so the rest of the finder is untouched.
"""

from __future__ import annotations

from typing import Optional

from ..models import SummaryChunk
from .store import load_chunks, load_chunks_for_product_code
from grounded_rag.retrieve import DEFAULT_STOPWORDS, RetrievalConfig, rank

# Sections that contain performance data get a score boost.
_SECTION_BOOST = {
    "Performance Testing": 2.0,
    "Conclusions / Limitations": 1.5,
    "Intended Use / Device Description": 1.2,
    "Substantial Equivalence": 1.1,
}

# IVD domain terms: strong signals when matched, kept even when short.
_DOMAIN_TERMS = {
    "lod": 3.0, "ppa": 3.0, "npa": 3.0, "loq": 3.0,
    "sensitivity": 2.0, "specificity": 2.0, "reactivity": 2.0,
    "predicate": 2.0, "comparator": 2.0, "reference": 1.5,
    "precision": 1.8, "reproducibility": 1.8, "detection": 1.5,
    "agreement": 1.5, "cross": 1.5, "interference": 1.8, "strain": 1.5,
    "limit": 1.3, "accuracy": 1.5, "cfu": 2.0, "copies": 1.5,
}

# Generic stopwords plus FDA-form noise that carries no retrieval signal.
_STOPWORDS = frozenset(DEFAULT_STOPWORDS | {
    "device", "test", "tests", "summary", "510", "k",
})

FDA_510K_RETRIEVAL = RetrievalConfig(
    section_boost=_SECTION_BOOST,
    domain_terms=_DOMAIN_TERMS,
    stopwords=_STOPWORDS,
)


def gather_candidates(
    k_numbers: Optional[list[str]] = None,
    product_codes: Optional[list[str]] = None,
) -> list[SummaryChunk]:
    """
    Gather the in-scope SummaryChunks (unranked) for a K-number / product-code
    scope. If neither is given, walks the whole index (slow on large corpora).

    Shared by retrieve() and the FDA-510(k) Corpus adapter so scoping logic lives
    in exactly one place.
    """
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

    if not k_numbers and not product_codes:
        # Full corpus scan — walk all chunk files.
        from ..index import store as _store
        for k in _store.list_indexed():
            candidates.extend(load_chunks(k))

    return candidates


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
    candidates = gather_candidates(k_numbers=k_numbers, product_codes=product_codes)
    return rank(query, candidates, FDA_510K_RETRIEVAL, top_k=top_k, sections=sections)
