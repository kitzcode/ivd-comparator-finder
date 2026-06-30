"""
The FDA 510(k) Corpus: candidates + retrieval config + grounding contract.

Layering: this adapter sits ABOVE the finder substrate and BELOW the engine.
  - It pulls candidate chunks from finder's index (finder.index),
  - reuses finder's IVD scoring config (FDA_510K_RETRIEVAL),
  - and owns the FDA grounding contract (predicate vs comparator, K-number
    citations, the refusal phrase).

The engine (grounded_rag) depends only on the Corpus protocol, never on this.
"""

from __future__ import annotations

from typing import Any, Optional

from finder.models import SummaryChunk
from finder.index.retrieve import FDA_510K_RETRIEVAL, gather_candidates

from grounded_rag.models import Chunk
from grounded_rag.retrieve import RetrievalConfig
from grounded_rag.contract import GroundingContract
from grounded_rag.corpus import RetrieveMixin

# ---------------------------------------------------------------------------
# FDA 510(k) grounding contract
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise scientific assistant answering questions about FDA 510(k) \
clearance summaries for in vitro diagnostic (IVD) devices.

You are given NUMBERED context candidates. RULES, violating any rule is a failure:
1. Answer ONLY from the numbered candidates. Do not use your training knowledge.
2. Cite every claim and every performance figure (PPA, NPA, sensitivity, \
   specificity, LoD, reactivity, etc.) with a bracketed candidate index like [2].
3. NEVER write a K-number, product code, device name as an identifier, accession, \
   or URL. The system attaches the real identifiers from the candidate you cite.
4. Distinguish clearly between:
   - The PREDICATE device (the legally marketed device cited for substantial equivalence)
   - The REFERENCE / COMPARATOR method (what performance was measured against)
   Conflating these two roles is an error.
5. If the candidates do not support an answer, respond exactly: \
   "The provided summaries do not contain sufficient information to answer this question."
6. After your answer, output a final line: SUPPORTING: [the indices you relied on].
"""

FDA_510K_CONTRACT = GroundingContract(
    system_prompt=_SYSTEM_PROMPT,
    not_found_sentinel="do not contain sufficient information",
    id_leak_pattern=r"(K\d{6}|DEN\d{6}|P\d{6})",
)


# ---------------------------------------------------------------------------
# SummaryChunk -> generic Chunk bridge
# ---------------------------------------------------------------------------

def summary_chunk_to_chunk(c: SummaryChunk) -> Chunk:
    """Map an FDA SummaryChunk onto the engine's generic Chunk. The K-number is
    both the doc_id (for citation matching) and the human label; the product code
    rides along in metadata, which the engine ignores."""
    return Chunk(
        doc_id=c.k_number,
        source_url=c.source_url,
        section=c.section,
        text=c.text,
        page=c.page,
        label=c.k_number,
        metadata={"product_code": c.product_code},
    )


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

class FDA510kCorpus(RetrieveMixin):
    """grounded_rag.Corpus over FDA 510(k) Summary chunks.

    Scope keys: ``k_numbers`` and/or ``product_codes`` (both optional; empty
    scope walks the whole index).
    """

    name = "fda_510k"

    def candidates(self, scope: Optional[dict[str, Any]] = None) -> list[Chunk]:
        scope = scope or {}
        summary_chunks = gather_candidates(
            k_numbers=scope.get("k_numbers"),
            product_codes=scope.get("product_codes"),
        )
        return [summary_chunk_to_chunk(c) for c in summary_chunks]

    @property
    def retrieval_config(self) -> RetrievalConfig:
        return FDA_510K_RETRIEVAL

    @property
    def grounding(self) -> GroundingContract:
        return FDA_510K_CONTRACT
