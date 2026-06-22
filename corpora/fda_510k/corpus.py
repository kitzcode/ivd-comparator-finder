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

# ---------------------------------------------------------------------------
# FDA 510(k) grounding contract
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise scientific assistant answering questions about FDA 510(k) \
clearance summaries for in vitro diagnostic (IVD) devices.

RULES — violating any rule is a failure:
1. Answer ONLY from the provided context chunks. Do not use your training knowledge.
2. Every performance figure (PPA, NPA, sensitivity, specificity, LoD, \
   reactivity, etc.) must be cited with the K-number and page number from the \
   chunk it came from.
3. If the answer is not in the chunks, respond: \
   "The provided summaries do not contain sufficient information to answer this question."
4. Distinguish clearly between:
   - The PREDICATE device (the legally marketed device cited for substantial equivalence)
   - The REFERENCE / COMPARATOR method (what performance was measured against)
   Conflating these two roles is an error.
5. Do not invent K-numbers, product codes, device names, or numeric values.
6. If a figure is ambiguous or the chunk is unclear, flag it rather than reporting a clean number.
"""

_CONTEXT_TEMPLATE = """\
--- CHUNK {i} | K-number: {doc_id} | Section: {section} | Page: {page} ---
{text}
"""

_USER_TEMPLATE = """\
Context chunks from 510(k) Summaries:

{context}

Question: {question}

Answer (cite K-number and page for every figure):
"""

FDA_510K_CONTRACT = GroundingContract(
    system_prompt=_SYSTEM_PROMPT,
    not_found_sentinel="do not contain sufficient information",
    cited_id_pattern=r"K\d{6}",
    context_template=_CONTEXT_TEMPLATE,
    user_template=_USER_TEMPLATE,
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

class FDA510kCorpus:
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
