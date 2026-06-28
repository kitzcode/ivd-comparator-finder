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
from grounded_rag.corpus import RetrieveMixin

from . import store

# ---------------------------------------------------------------------------
# Grounding contract
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise regulatory assistant answering questions strictly from FDA \
guidance documents.

You are given NUMBERED context candidates. RULES, violating any rule is a failure:
1. Answer ONLY from the numbered candidates. Do not use your training knowledge.
2. Cite every statement of FDA recommendation or policy with a bracketed candidate \
   index like [2].
3. NEVER write a guidance document tag, accession, citation, or URL. The system \
   attaches the real identifiers from the candidate you cite.
4. Guidance is nonbinding; describe it as FDA's current thinking / recommendations, \
   not as binding requirements, unless the candidate text itself cites a regulation \
   (e.g. 21 CFR).
5. If the candidates do not support an answer, respond exactly: \
   "The provided guidance documents do not contain sufficient information to answer this question."
6. After your answer, output a final line: SUPPORTING: [the indices you relied on].
"""

FDA_GUIDANCE_CONTRACT = GroundingContract(
    system_prompt=_SYSTEM_PROMPT,
    not_found_sentinel="do not contain sufficient information",
    id_leak_pattern=r"FDA-GUID-\d+",
)

# Generic retrieval config: default English stopwords, no domain-term boosts.
FDA_GUIDANCE_RETRIEVAL = RetrievalConfig()


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

class FDAGuidanceCorpus(RetrieveMixin):
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
