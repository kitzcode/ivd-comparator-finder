"""
Generic keyword retrieval and scoring. No embeddings, no FDA knowledge.

Scoring is identical in spirit to the original finder retriever, but the
domain-term weights, section boosts, and stopwords are supplied per corpus via
`RetrievalConfig` rather than hard-coded. The scorer is structural: it ranks any
object exposing `.text: str` and `.section: str`, so callers get their own chunk
type back (e.g. the FDA `SummaryChunk`), not a converted copy.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Protocol, TypeVar, runtime_checkable

# Common English words that carry no retrieval signal.
DEFAULT_STOPWORDS: set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "has", "have", "how", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "their", "this", "to", "was", "were", "what", "when",
    "which", "who", "with", "your", "you", "i", "me", "my", "we", "us",
}


@runtime_checkable
class ChunkLike(Protocol):
    """Anything the scorer can rank: needs text and a section label."""

    text: str
    section: str


C = TypeVar("C", bound=ChunkLike)


@dataclass(frozen=True)
class RetrievalConfig:
    """Per-corpus scoring knobs.

    section_boost: multiply a chunk's score when its section matches (favour
        sections that carry the answer, e.g. performance data).
    domain_terms: terms that are strong signals when matched, with weights;
        also kept as query terms even if short (e.g. "lod", "ppa").
    stopwords: query terms to drop entirely.
    """

    section_boost: dict[str, float] = field(default_factory=dict)
    domain_terms: dict[str, float] = field(default_factory=dict)
    stopwords: frozenset[str] = field(default_factory=lambda: frozenset(DEFAULT_STOPWORDS))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def query_terms(query: str, config: RetrievalConfig) -> list[str]:
    """Tokenize a query, dropping stopwords but keeping short domain terms."""
    terms: list[str] = []
    for t in tokenize(query):
        if t in config.domain_terms:
            terms.append(t)
        elif t not in config.stopwords and len(t) > 1:
            terms.append(t)
    return terms


def score_chunk(chunk: ChunkLike, terms: list[str], config: RetrievalConfig) -> float:
    """Score a chunk against query terms.

    - Term frequency with diminishing returns: 1 + ln(count) per matched term.
    - Domain terms weighted higher.
    - Normalized by sqrt(distinct tokens) so long boilerplate doesn't win by size.
    - Coverage bonus rewards hitting more distinct query terms.
    - Section boost favours answer-bearing sections.
    """
    counts = Counter(tokenize(chunk.text))
    if not counts:
        return 0.0

    raw = 0.0
    matched_terms = 0
    for term in terms:
        c = counts.get(term, 0)
        if c:
            matched_terms += 1
            weight = config.domain_terms.get(term, 1.0)
            raw += weight * (1.0 + math.log(c))

    if matched_terms == 0:
        return 0.0

    coverage = matched_terms / max(len(terms), 1)
    raw *= (0.5 + coverage)

    length_norm = math.sqrt(max(len(counts), 1))
    score = (raw / length_norm) * config.section_boost.get(chunk.section, 1.0)
    return score


def rank(
    query: str,
    candidates: list[C],
    config: RetrievalConfig,
    top_k: int = 8,
    sections: Optional[list[str]] = None,
) -> list[C]:
    """Return the top_k candidates ranked by relevance to query.

    Preserves the candidate type: whatever objects go in come back out.
    Pass `sections` to restrict to specific section labels (case-insensitive).
    """
    if sections:
        section_set = {s.lower() for s in sections}
        candidates = [c for c in candidates if c.section.lower() in section_set]

    if not candidates:
        return []

    terms = query_terms(query, config)
    if not terms:
        return candidates[:top_k]

    scored = [(c, score_chunk(c, terms, config)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)

    nonzero = [(c, s) for c, s in scored if s > 0]
    result_pairs = nonzero if nonzero else scored
    return [c for c, _ in result_pairs[:top_k]]


def best_snippet(question: str, text: str, config: RetrievalConfig, window: int = 600) -> str:
    """Extract the most relevant passage from a chunk for a keyword query.

    Scores each sentence/line by query-term hits and grows a window around the
    best one up to the char budget, so the user sees the relevant fragment, not
    a wall of text. With no usable terms, returns the head of the text.
    """
    terms = set(query_terms(question, config))
    if not terms:
        return text[:window].strip()

    fragments = [f.strip() for f in re.split(r"(?<=[.;:])\s+|\n+", text) if f.strip()]
    if not fragments:
        return text[:window].strip()

    best_idx, best_score = 0, -1
    for i, frag in enumerate(fragments):
        ftokens = set(tokenize(frag))
        score = len(terms & ftokens)
        if score > best_score:
            best_idx, best_score = i, score

    if best_score <= 0:
        return text[:window].strip()

    chosen = [fragments[best_idx]]
    lo, hi = best_idx - 1, best_idx + 1
    total = len(fragments[best_idx])
    while total < window and (lo >= 0 or hi < len(fragments)):
        if lo >= 0:
            chosen.insert(0, fragments[lo]); total += len(fragments[lo]); lo -= 1
        if hi < len(fragments) and total < window:
            chosen.append(fragments[hi]); total += len(fragments[hi]); hi += 1
    snippet = " ".join(chosen).strip()
    return (snippet[:window] + "…") if len(snippet) > window else snippet
