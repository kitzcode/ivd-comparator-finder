"""
M3: Grounded Q&A over indexed 510(k) Summary chunks.

Contract:
  - Every claim in the answer must trace to a retrieved chunk.
  - If retrieved chunks don't answer the question, say so explicitly.
  - Performance figures are extracted from chunks and cited by K-number + page.
  - Predicate device and reference/comparator method are always kept distinct.
  - Model memory is never used to fill a gap.

This module is intentionally model-agnostic. Pass any callable that takes
(system_prompt: str, user_prompt: str) -> str as the `llm` argument.
The caller is responsible for supplying and paying for the LLM.

If no LLM is supplied, the module falls back to a keyword-extraction mode
that returns the most relevant chunk text verbatim with a citation — no
generation, fully grounded.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from .models import Answer, Citation, SummaryChunk
from .index.retrieve import retrieve

# ---------------------------------------------------------------------------
# Grounding contract prompts
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
--- CHUNK {i} | K-number: {k} | Section: {section} | Page: {page} ---
{text}
"""

_USER_TEMPLATE = """\
Context chunks from 510(k) Summaries:

{context}

Question: {question}

Answer (cite K-number and page for every figure):
"""


# ---------------------------------------------------------------------------
# Keyword-only fallback (no LLM)
# ---------------------------------------------------------------------------

def _keyword_answer(question: str, chunks: list[SummaryChunk]) -> Answer:
    """
    Return the top chunk's text verbatim with a citation.
    No generation — fully grounded, no hallucination risk.
    """
    if not chunks:
        return Answer(
            question=question,
            answer="",
            not_found_reason="No indexed summaries were found for this query scope.",
        )

    top = chunks[0]
    return Answer(
        question=question,
        answer=top.text,
        citations=[Citation(
            k_number=top.k_number,
            source_url=top.source_url,
            page=top.page,
            section=top.section,
        )],
    )


# ---------------------------------------------------------------------------
# LLM-backed answer
# ---------------------------------------------------------------------------

def _llm_answer(
    question: str,
    chunks: list[SummaryChunk],
    llm: Callable[[str, str], str],
) -> Answer:
    if not chunks:
        return Answer(
            question=question,
            answer="",
            not_found_reason="No indexed summaries were found for this query scope.",
        )

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(_CONTEXT_TEMPLATE.format(
            i=i,
            k=chunk.k_number,
            section=chunk.section,
            page=chunk.page or "?",
            text=chunk.text,
        ))
    context = "\n".join(context_parts)
    user_prompt = _USER_TEMPLATE.format(context=context, question=question)

    raw_answer = llm(_SYSTEM_PROMPT, user_prompt)

    # Extract citations from the answer text (K-numbers mentioned)
    cited_k = set(re.findall(r"K\d{6}", raw_answer))
    citations = [
        Citation(
            k_number=c.k_number,
            source_url=c.source_url,
            page=c.page,
            section=c.section,
        )
        for c in chunks
        if c.k_number in cited_k
    ]
    # Deduplicate citations by k_number+page
    seen = set()
    unique_citations: list[Citation] = []
    for cit in citations:
        key = (cit.k_number, cit.page)
        if key not in seen:
            seen.add(key)
            unique_citations.append(cit)

    not_found = None
    if "do not contain sufficient information" in raw_answer.lower():
        not_found = raw_answer

    return Answer(
        question=question,
        answer=raw_answer if not not_found else "",
        citations=unique_citations,
        not_found_reason=not_found,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask(
    question: str,
    k_numbers: Optional[list[str]] = None,
    product_codes: Optional[list[str]] = None,
    top_k: int = 8,
    sections: Optional[list[str]] = None,
    llm: Optional[Callable[[str, str], str]] = None,
) -> Answer:
    """
    Answer a question about 510(k) Summary content.

    Scope with k_numbers and/or product_codes. If neither is given, searches
    the entire index (slow on large corpora).

    Pass an `llm` callable for generated answers; omit for keyword-only mode
    (returns the top chunk verbatim — fully grounded, no generation).

    The llm callable signature: (system_prompt: str, user_prompt: str) -> str
    """
    chunks = retrieve(
        question,
        k_numbers=k_numbers,
        product_codes=product_codes,
        top_k=top_k,
        sections=sections,
    )

    if llm is None:
        return _keyword_answer(question, chunks)
    return _llm_answer(question, chunks, llm)


def format_answer(answer: Answer) -> str:
    """Pretty-print an Answer for CLI output."""
    lines: list[str] = []

    if answer.not_found_reason:
        lines.append(f"NOT FOUND: {answer.not_found_reason}")
    elif answer.answer:
        lines.append(answer.answer)
    else:
        lines.append("[No answer generated]")

    if answer.citations:
        lines.append("\nCitations:")
        for cit in answer.citations:
            page_str = f" p.{cit.page}" if cit.page else ""
            section_str = f" [{cit.section}]" if cit.section else ""
            lines.append(f"  {cit.k_number}{page_str}{section_str}  {cit.source_url}")

    return "\n".join(lines)
