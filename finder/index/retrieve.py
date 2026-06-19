"""
Retrieval: find the most relevant SummaryChunks for a query.

v2 strategy: section-targeted keyword search (no embeddings).
- Pre-filter: scope to the requested K-number(s) or product code(s).
- Score: count query term hits in each chunk, weight by section relevance.
- Return the top-k chunks, ranked by score.

Embeddings are deferred until keyword retrieval proves insufficient.
"""

from __future__ import annotations

import math
import re
from collections import Counter
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

# Common English words that carry no retrieval signal. Filtered from the query
# so a chunk can't win just by containing "what / is / the".
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "has", "have", "how", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "their", "this", "to", "was", "were", "what", "when",
    "which", "who", "with", "your", "you", "i", "me", "my", "we", "us",
    "device", "test", "tests", "summary", "510", "k",
}

# Domain terms that are strong signals when they match — weighted higher.
_DOMAIN_TERMS = {
    "lod": 3.0, "ppa": 3.0, "npa": 3.0, "loq": 3.0,
    "sensitivity": 2.0, "specificity": 2.0, "reactivity": 2.0,
    "predicate": 2.0, "comparator": 2.0, "reference": 1.5,
    "precision": 1.8, "reproducibility": 1.8, "detection": 1.5,
    "agreement": 1.5, "cross": 1.5, "interference": 1.8, "strain": 1.5,
    "limit": 1.3, "accuracy": 1.5, "cfu": 2.0, "copies": 1.5,
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _query_terms(query: str) -> list[str]:
    """Tokenize a query, dropping stopwords but keeping short domain terms."""
    terms = []
    for t in _tokenize(query):
        if t in _DOMAIN_TERMS:
            terms.append(t)
        elif t not in _STOPWORDS and len(t) > 1:
            terms.append(t)
    return terms


def _score_chunk(chunk: SummaryChunk, query_terms: list[str]) -> float:
    """
    Score a chunk against the query terms.

    - Term frequency with diminishing returns: 1 + ln(count) per matched term,
      so a term appearing 10x doesn't swamp coverage of distinct terms.
    - Domain terms (lod, ppa, ...) are weighted higher.
    - Normalized by sqrt(length) so long boilerplate chunks don't win by size.
    - Section boost favours performance-bearing sections.
    """
    counts = Counter(_tokenize(chunk.text))
    if not counts:
        return 0.0

    raw = 0.0
    matched_terms = 0
    for term in query_terms:
        c = counts.get(term, 0)
        if c:
            matched_terms += 1
            weight = _DOMAIN_TERMS.get(term, 1.0)
            raw += weight * (1.0 + math.log(c))

    if matched_terms == 0:
        return 0.0

    # Coverage bonus: reward chunks that hit more distinct query terms.
    coverage = matched_terms / max(len(query_terms), 1)
    raw *= (0.5 + coverage)

    # Length normalization (sqrt damping keeps it gentle).
    length_norm = math.sqrt(max(len(counts), 1))
    score = (raw / length_norm) * _SECTION_BOOST.get(chunk.section, 1.0)
    return score


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

    query_terms = _query_terms(query)
    if not query_terms:
        return candidates[:top_k]

    scored = [(c, _score_chunk(c, query_terms)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Filter out zero-score chunks unless there's nothing better
    nonzero = [(c, s) for c, s in scored if s > 0]
    result_pairs = nonzero if nonzero else scored
    return [c for c, _ in result_pairs[:top_k]]
