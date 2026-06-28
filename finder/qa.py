"""
FDA 510(k) grounded Q&A — now a thin shim over the engine + corpus.

finder.qa.ask() routes through grounded_rag.qa.ask_corpus() against the
FDA510kCorpus, then maps the engine's generic Answer back to the FDA Answer type
(citations keyed by K-number) so the public surface is unchanged.

All grounding behaviour (refusal gate, keyword fallback, "the model never writes
a citation") lives in grounded_rag; the FDA framing lives in corpora.fda_510k.

Pass any callable (system_prompt, user_prompt) -> str as `llm`. Omit it for
keyword-only mode (top chunk verbatim with a citation, fully grounded).
"""

from __future__ import annotations

from typing import Callable, Optional

from .models import Answer, Citation

from grounded_rag.qa import ask_corpus
from corpora.fda_510k import FDA510kCorpus, FDA_510K_CONTRACT  # noqa: F401 (re-export)

_CORPUS = FDA510kCorpus()


def _to_answer(generic) -> Answer:
    """Map a grounded_rag.Answer back to the FDA Answer (Citation keyed by K-number)."""
    return Answer(
        question=generic.question,
        answer=generic.answer,
        citations=[
            Citation(
                k_number=c.doc_id,
                source_url=c.source_url,
                page=c.page,
                section=c.section,
                snippet=c.snippet,
            )
            for c in generic.citations
        ],
        not_found_reason=generic.not_found_reason,
    )


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
    generic = ask_corpus(
        _CORPUS,
        question,
        scope={"k_numbers": k_numbers, "product_codes": product_codes},
        top_k=top_k,
        sections=sections,
        llm=llm,
    )
    return _to_answer(generic)


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
