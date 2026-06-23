"""
The grounded-answer machinery, corpus-agnostic.

Two modes, both grounded:
  - keyword (no LLM): return the most relevant passage of the top chunk verbatim
    with a citation. No generation, so no hallucination is possible.
  - LLM-backed: pass the retrieved chunks as context under the corpus's grounding
    contract. The model writes prose; the core reconstructs citations from the
    chunks the model referenced and detects the refusal sentinel. The model never
    emits a Citation object itself.

`answer()` operates on already-retrieved Chunks so retrieval and reasoning stay
separable. `ask_corpus()` is the convenience that wires a Corpus end to end.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from .models import Answer, Chunk, Citation
from .contract import GroundingContract
from .retrieve import RetrievalConfig, best_snippet, rank

# An LLM is any callable (system_prompt, user_prompt) -> str. The caller supplies
# and pays for it; the core stays model-agnostic.
LLM = Callable[[str, str], str]

_NO_SOURCES = "No sources were found for this query scope."


def _cite(chunk: Chunk) -> Citation:
    return Citation(
        doc_id=chunk.doc_id,
        source_url=chunk.source_url,
        page=chunk.page,
        section=chunk.section,
        label=chunk.label,
    )


def _keyword_answer(question: str, chunks: list[Chunk], config: RetrievalConfig) -> Answer:
    if not chunks:
        return Answer(question=question, answer="", not_found_reason=_NO_SOURCES)
    top = chunks[0]
    return Answer(
        question=question,
        answer=best_snippet(question, top.text, config),
        citations=[_cite(top)],
    )


def _cited_chunks(raw: str, chunks: list[Chunk], contract: GroundingContract) -> list[Citation]:
    """Reconstruct citations from the chunks the model actually referenced.

    If the contract supplies an id pattern, match ids the model emitted against
    chunk doc_ids. Otherwise treat a chunk as cited when its doc_id appears
    literally in the answer. Deduplicated by (doc_id, page)."""
    if contract.cited_id_pattern:
        cited_ids = set(re.findall(contract.cited_id_pattern, raw))
        referenced = [c for c in chunks if c.doc_id in cited_ids]
    else:
        referenced = [c for c in chunks if c.doc_id and c.doc_id in raw]

    seen: set[tuple[str, Optional[int]]] = set()
    out: list[Citation] = []
    for c in referenced:
        key = (c.doc_id, c.page)
        if key not in seen:
            seen.add(key)
            out.append(_cite(c))
    return out


def _llm_answer(
    question: str,
    chunks: list[Chunk],
    llm: LLM,
    contract: GroundingContract,
) -> Answer:
    if not chunks:
        return Answer(question=question, answer="", not_found_reason=_NO_SOURCES)

    context = "\n".join(
        contract.context_template.format(
            i=i,
            doc_id=chunk.doc_id,
            section=chunk.section,
            page=chunk.page if chunk.page is not None else "?",
            text=chunk.text,
        )
        for i, chunk in enumerate(chunks, 1)
    )
    user_prompt = contract.user_template.format(context=context, question=question)
    raw = llm(contract.system_prompt, user_prompt)

    not_found = raw if contract.not_found_sentinel.lower() in raw.lower() else None
    return Answer(
        question=question,
        answer="" if not_found else raw,
        citations=[] if not_found else _cited_chunks(raw, chunks, contract),
        not_found_reason=not_found,
    )


def answer(
    question: str,
    chunks: list[Chunk],
    *,
    contract: GroundingContract,
    config: RetrievalConfig,
    llm: Optional[LLM] = None,
) -> Answer:
    """Answer `question` from already-retrieved `chunks`.

    Omit `llm` for keyword-only mode (top chunk verbatim, fully grounded).
    Supply an `llm` callable for a generated answer under the grounding contract.
    """
    if llm is None:
        return _keyword_answer(question, chunks, config)
    return _llm_answer(question, chunks, llm, contract)


def ask_corpus(
    corpus,
    question: str,
    *,
    scope: Optional[dict[str, Any]] = None,
    top_k: int = 8,
    sections: Optional[list[str]] = None,
    llm: Optional[LLM] = None,
) -> Answer:
    """End-to-end convenience: gather candidates from a Corpus, rank, and answer."""
    candidates = corpus.candidates(scope)
    chunks = rank(question, candidates, corpus.retrieval_config, top_k=top_k, sections=sections)
    return answer(
        question,
        chunks,
        contract=corpus.grounding,
        config=corpus.retrieval_config,
        llm=llm,
    )
