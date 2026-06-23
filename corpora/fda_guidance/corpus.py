"""
The FDA guidance Corpus: the SAME grounded_rag engine pointed at a second body
of documents. This is the generalization proof.

What differs from the 510(k) corpus is only data and contract:
  - candidates come from the guidance chunk store (generic Chunks already),
  - the grounding contract cites by guidance tag (FDA-GUID-NNNNN) + page and
    frames the task as answering from guidance text,
  - the retrieval config is left generic (guidance Q&A has no single
    answer-bearing section the way performance data does in a 510(k)).
"""

from __future__ import annotations

from typing import Any, Optional

from grounded_rag.models import Chunk
from grounded_rag.retrieve import RetrievalConfig
from grounded_rag.contract import GroundingContract

from . import store

# ---------------------------------------------------------------------------
# Grounding contract
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise regulatory assistant answering questions strictly from FDA \
guidance documents.

RULES — violating any rule is a failure:
1. Answer ONLY from the provided context chunks. Do not use your training knowledge.
2. Every statement of FDA recommendation or policy must be cited with the guidance \
   document tag (e.g. FDA-GUID-71075) and page number from the chunk it came from.
3. Guidance is nonbinding; describe it as FDA's current thinking / recommendations, \
   not as binding requirements, unless the text itself cites a regulation (e.g. 21 CFR).
4. If the answer is not in the chunks, respond: \
   "The provided guidance documents do not contain sufficient information to answer this question."
5. Do not invent document tags, citations, dates, or regulatory section numbers.
"""

_CONTEXT_TEMPLATE = """\
--- CHUNK {i} | Guidance: {doc_id} | Section: {section} | Page: {page} ---
{text}
"""

_USER_TEMPLATE = """\
Context chunks from FDA guidance documents:

{context}

Question: {question}

Answer (cite the guidance tag and page for every statement):
"""

FDA_GUIDANCE_CONTRACT = GroundingContract(
    system_prompt=_SYSTEM_PROMPT,
    not_found_sentinel="do not contain sufficient information",
    cited_id_pattern=r"FDA-GUID-\d+",
    context_template=_CONTEXT_TEMPLATE,
    user_template=_USER_TEMPLATE,
)

# Generic retrieval config: default English stopwords, no domain-term boosts.
FDA_GUIDANCE_RETRIEVAL = RetrievalConfig()


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

class FDAGuidanceCorpus:
    """grounded_rag.Corpus over FDA guidance documents.

    Scope key: ``doc_ids`` (list of "FDA-GUID-NNNNN"). Empty scope = whole corpus.
    """

    name = "fda_guidance"

    def candidates(self, scope: Optional[dict[str, Any]] = None) -> list[Chunk]:
        scope = scope or {}
        doc_ids = scope.get("doc_ids")
        if doc_ids:
            out: list[Chunk] = []
            for d in doc_ids:
                out.extend(store.load_chunks(d))
            return out
        return store.load_all_chunks()

    @property
    def retrieval_config(self) -> RetrievalConfig:
        return FDA_GUIDANCE_RETRIEVAL

    @property
    def grounding(self) -> GroundingContract:
        return FDA_GUIDANCE_CONTRACT
