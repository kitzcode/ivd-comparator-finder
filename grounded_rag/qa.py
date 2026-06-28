"""
The grounded-answer machinery, corpus-agnostic.

Two modes, both grounded, both leakage-free:

  - keyword (no LLM): return the most relevant passage of the top chunk verbatim
    with a code-attached citation. No generation, so no hallucination is possible.

  - LLM-backed (index-based selection): the model sees the retrieved candidates
    NUMBERED, by index only, with their doc ids and URLs withheld. It answers using
    [n] markers and returns the indices it relied on. Code then attaches the real
    SourceRef (doc_id, source_url, page, snippet) for each selected index. The model
    never writes an identifier or a URL. Three guards enforce the contract:
      1. leakage guard  — if the model's prose contains an identifier (per the
         contract's id_leak_pattern) or a URL, the answer is blanked and refused.
      2. source-existence — indices outside the retrieved set are dropped, so the
         model cannot cite a candidate that was not retrieved.
      3. refusal gate    — the not_found sentinel, or no usable index, returns a
         refusal object rather than an unsupported answer.

`answer()` operates on already-retrieved Chunks so retrieval and reasoning stay
separable. `ask_corpus()` wires a Corpus end to end.
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
_UNSUPPORTED = "The retrieved sources do not support an answer to this question."
_LEAK_REFUSED = (
    "Answer withheld: the model emitted an identifier or URL, which violates the "
    "grounding contract (citations must be attached by code, not written by the model)."
)

_URL = re.compile(r"https?://", re.I)
_SUPPORTING = re.compile(r"SUPPORTING:\s*\[([^\]]*)\]", re.I)
_INLINE_IDX = re.compile(r"\[(\d+)\]")


def _cite(chunk: Chunk, snippet: Optional[str] = None) -> Citation:
    return Citation(
        doc_id=chunk.doc_id,
        source_url=chunk.source_url,
        page=chunk.page,
        section=chunk.section,
        label=chunk.label,
        snippet=snippet,
    )


def _keyword_answer(question: str, chunks: list[Chunk], config: RetrievalConfig) -> Answer:
    if not chunks:
        return Answer(question=question, answer="", not_found_reason=_NO_SOURCES)
    top = chunks[0]
    snippet = best_snippet(question, top.text, config)
    return Answer(question=question, answer=snippet, citations=[_cite(top, snippet)])


def _selected_indices(raw: str, n: int) -> list[int]:
    """Return the 1-based indices the model relied on, validated to 1..n, in first
    appearance order, deduped. Combines the SUPPORTING: line and inline [n] markers.
    Out-of-range indices are dropped (source-existence guard)."""
    ordered: list[int] = []
    seen: set[int] = set()

    def _add(tokens: list[str]) -> None:
        for tok in tokens:
            tok = tok.strip()
            # ASCII digits only: str.isdigit() accepts Unicode digit lookalikes
            # (e.g. fullwidth or mathematical digits) that int() would still parse.
            if not re.fullmatch(r"[0-9]+", tok):
                continue
            idx = int(tok)
            if 1 <= idx <= n and idx not in seen:
                seen.add(idx)
                ordered.append(idx)

    m = _SUPPORTING.search(raw)
    if m:
        _add(m.group(1).split(","))
    _add(_INLINE_IDX.findall(raw))
    return ordered


def _strip_supporting(raw: str) -> str:
    """Remove the trailing SUPPORTING: line from the visible answer."""
    return _SUPPORTING.sub("", raw).strip()


def _has_leak(prose: str, contract: GroundingContract) -> bool:
    if _URL.search(prose):
        return True
    # Case-insensitive so a lowercased identifier (k173653) cannot slip the guard.
    if contract.id_leak_pattern and re.search(contract.id_leak_pattern, prose, re.I):
        return True
    return False


def _llm_answer(
    question: str,
    chunks: list[Chunk],
    llm: LLM,
    contract: GroundingContract,
    config: RetrievalConfig,
) -> Answer:
    if not chunks:
        return Answer(question=question, answer="", not_found_reason=_NO_SOURCES)

    # Show candidates by index only. doc_id and source_url are withheld.
    context = "\n".join(
        contract.context_template.format(
            i=i,
            section=chunk.section,
            page=chunk.page if chunk.page is not None else "?",
            text=chunk.text,
        )
        for i, chunk in enumerate(chunks, 1)
    )
    user_prompt = contract.user_template.format(context=context, question=question)
    raw = llm(contract.system_prompt, user_prompt)

    # Refusal sentinel.
    if contract.not_found_sentinel.lower() in raw.lower():
        return Answer(question=question, answer="", not_found_reason=raw.strip())

    # Leakage guard runs on the FULL raw output, before the SUPPORTING line is
    # stripped, so an identifier hidden in that line cannot bypass the guard.
    if _has_leak(raw, contract):
        return Answer(question=question, answer="", not_found_reason=_LEAK_REFUSED)

    # Source-existence: only indices into the retrieved set survive.
    indices = _selected_indices(raw, len(chunks))
    if not indices:
        return Answer(question=question, answer="", not_found_reason=_UNSUPPORTED)

    prose = _strip_supporting(raw)

    citations = [
        _cite(chunks[i - 1], best_snippet(question, chunks[i - 1].text, config))
        for i in indices
    ]
    return Answer(question=question, answer=prose, citations=citations)


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
    Supply an `llm` callable for index-based generation under the grounding contract.
    """
    if llm is None:
        return _keyword_answer(question, chunks, config)
    return _llm_answer(question, chunks, llm, contract, config)


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
