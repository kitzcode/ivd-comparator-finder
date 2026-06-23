"""
Grounded-RAG MCP — the REASONING layer.

Read-only, typed tools that run the corpus-agnostic engine over any registered
corpus. The same `ask` tool answers questions about 510(k) Summaries AND FDA
guidance documents; the corpus is chosen by name. Keyword-only mode (no LLM
generation) so answers are returned verbatim from source with citations.

Tools (readOnlyHint=True, destructiveHint=False):
  list_corpora         enumerate registered corpora and their scope keys
  ask                  grounded Q&A over a chosen corpus (keyword mode)
  compare_performance  structured 510(k) performance extraction (PPA/NPA/LoD/...)

Run:
  python -m mcp_servers.grounded_rag
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP(
    "grounded-rag",
    instructions=(
        "Grounded-RAG reasoning layer. Answers questions strictly from retrieved "
        "source chunks, with a citation for every figure, and refuses when the "
        "sources do not support an answer. One engine, multiple corpora "
        "(510(k) Summaries and FDA guidance). The model never writes a citation; "
        "citations are reconstructed from the chunks actually used."
    ),
)

_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False)


@mcp.tool(
    annotations=_READ_ONLY,
    description=(
        "List the corpora available to ask(). Returns each corpus name and a "
        "description of its scope keys and citation style."
    ),
)
def list_corpora() -> dict:
    from corpora.registry import list_corpora as _list
    return {"corpora": _list()}


@mcp.tool(
    annotations=_READ_ONLY,
    description=(
        "Answer a question grounded in a chosen corpus, keyword-only (no LLM "
        "generation): returns the most relevant chunk verbatim with a citation, or "
        "a not_found reason when the sources do not support an answer. "
        "corpus='fda_510k' scopes with k_numbers/product_codes (cited by K-number); "
        "corpus='fda_guidance' scopes with doc_ids (cited by guidance tag). "
        "Empty cells / not_found mean the data was not in the indexed sources, "
        "not that it does not exist."
    ),
)
def ask(
    question: str,
    corpus: str = "fda_510k",
    k_numbers: list[str] | None = None,
    product_codes: list[str] | None = None,
    doc_ids: list[str] | None = None,
    top_k: int = 5,
) -> dict:
    """
    question: the question to answer
    corpus: 'fda_510k' or 'fda_guidance' (see list_corpora)
    k_numbers / product_codes: scope for the fda_510k corpus
    doc_ids: scope for the fda_guidance corpus (FDA-GUID-NNNNN)
    top_k: number of chunks to retrieve
    """
    from corpora.registry import get_corpus
    from grounded_rag.qa import ask_corpus

    try:
        c = get_corpus(corpus)
    except KeyError as e:
        return {"error": str(e)}

    scope = {
        "k_numbers": k_numbers,
        "product_codes": product_codes,
        "doc_ids": doc_ids,
    }
    answer = ask_corpus(c, question, scope=scope, top_k=top_k, llm=None)

    return {
        "corpus": corpus,
        "question": answer.question,
        "answer": answer.answer or None,
        "not_found_reason": answer.not_found_reason,
        "citations": [
            {
                "doc_id": cit.doc_id,
                "label": cit.label,
                "page": cit.page,
                "section": cit.section,
                "source_url": cit.source_url,
            }
            for cit in answer.citations
        ],
    }


@mcp.tool(
    annotations=_READ_ONLY,
    description=(
        "Extract structured performance data (PPA, NPA, LoD, reactivity strains, "
        "comparator/reference method, predicate device) from indexed 510(k) Summaries "
        "for a list of K-numbers. "
        "Every value carries a citation to K-number + page. "
        "Empty cells mean the data was not found in the indexed summary — not that "
        "it was not studied. "
        "PREDICATE ≠ COMPARATOR: the predicate device is cited for substantial equivalence; "
        "the comparator/reference method is what performance was measured against."
    ),
)
def compare_performance(k_numbers: list[str]) -> dict:
    """
    k_numbers: list of K-numbers to compare (must be indexed via ingest first)
    """
    from finder.extract import extract_performance
    from finder.sources.openfda import get_510k_by_knumber

    device_names: dict[str, str] = {}
    product_codes_map: dict[str, str] = {}
    for k in k_numbers:
        rec = get_510k_by_knumber(k)
        if rec:
            device_names[k] = rec.get("device_name", "")
            product_codes_map[k] = rec.get("product_code", "")

    table = extract_performance(k_numbers, device_names=device_names, product_codes=product_codes_map)

    def _pv(val) -> dict | None:
        if val is None:
            return None
        return {
            "value": val.value,
            "citation": {
                "k_number": val.citation.k_number,
                "page": val.citation.page,
                "section": val.citation.section,
                "source_url": val.citation.source_url,
            },
        }

    return {
        "scope_note": table.scope_note,
        "predicate_note": table.predicate_note,
        "rows": [
            {
                "k_number": row.k_number,
                "device_name": row.device_name,
                "product_code": row.product_code,
                "ppa": _pv(row.ppa),
                "npa": _pv(row.npa),
                "lod": _pv(row.lod),
                "comparator_method": _pv(row.comparator_method),
                "predicate_device": _pv(row.predicate_device),
                "reactivity_strains": _pv(row.reactivity_strains),
                "extraction_notes": row.extraction_notes,
            }
            for row in table.rows
        ],
    }


if __name__ == "__main__":
    mcp.run()
