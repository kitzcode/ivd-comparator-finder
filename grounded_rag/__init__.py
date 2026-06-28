"""
grounded_rag: a corpus-agnostic grounded-RAG reasoning core.

This is the reasoning layer, deliberately source-agnostic. It knows nothing
about FDA, K-numbers, or PDFs. It scores chunks against a query, retrieves the
most relevant, and answers from them under a strict grounding contract:

  - Every claim traces to a retrieved chunk.
  - The model never writes a citation; citations are reconstructed from the
    retrieved chunks the model actually referenced.
  - If retrieved chunks don't answer the question, refuse explicitly.
  - In keyword-only mode (no LLM), the top chunk's most relevant passage is
    returned verbatim with a citation — fully grounded, no generation.

A corpus plugs in by supplying:
  - candidate chunks for a query scope (the Corpus protocol),
  - a RetrievalConfig (domain term weights, section boosts, stopwords),
  - a GroundingContract (system prompt, citation id pattern, refusal sentinel).
"""

from __future__ import annotations

from .models import Chunk, Citation, Answer
from .retrieve import RetrievalConfig, rank, best_snippet, query_terms
from .contract import GroundingContract
from .qa import answer, ask_corpus
from .corpus import Corpus, RetrieveMixin

__all__ = [
    "Chunk",
    "Citation",
    "Answer",
    "RetrievalConfig",
    "rank",
    "best_snippet",
    "query_terms",
    "GroundingContract",
    "answer",
    "ask_corpus",
    "Corpus",
    "RetrieveMixin",
]
