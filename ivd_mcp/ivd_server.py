"""
M6: Thin read-only MCP server for the IVD Comparator Finder.

Tools (all readOnlyHint=True, destructiveHint=False):
  find_devices          analyte term -> device table (K-number, applicant, dates, product code, regulation)
  get_clearance         single K-number -> full clearance record + index status
  ask_summary           grounded Q&A over indexed 510(k) Summary chunks (keyword mode; no LLM)
  compare_performance   structured performance extraction table for a list of K-numbers
  find_reference_labs   lab test directory lookup (ARUP / Mayo; labeled as directory lookup)

Run:
  python -m mcp.server   (from repo root, via __main__.py)
  or:
  mcp dev mcp/server.py  (for interactive testing)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on the path when launched as a module
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP(
    "ivd-comparator-finder",
    instructions=(
        "IVD Predicate / Comparator Finder. "
        "All device data comes from openFDA. "
        "Performance data is extracted from 510(k) Summary PDFs with citations. "
        "Reference-lab results are directory lookups, not FDA determinations. "
        "Predicate device (substantial equivalence) ≠ comparator/reference method (performance study)."
    ),
)

_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False)

# ---------------------------------------------------------------------------
# Tool: find_devices
# ---------------------------------------------------------------------------

@mcp.tool(
    annotations=_READ_ONLY,
    description=(
        "Map an analyte or assay term to FDA-cleared IVD devices. "
        "Returns a table of K-numbers, device names, applicants, decision dates, "
        "product codes, and regulation numbers. "
        "Heuristic: uses synonym text search against openFDA; always surface the "
        "synonym set used so the caller can verify completeness."
    ),
)
def find_devices(
    analyte: str,
    extra_synonyms: list[str] | None = None,
    resolve_summary_urls: bool = False,
) -> dict:
    """
    analyte: analyte or assay name (e.g. 'Group A Strep', 'Streptococcus pyogenes')
    extra_synonyms: additional synonyms to include in the search
    resolve_summary_urls: if True, probe accessdata.fda.gov for each Summary PDF URL (slow)
    """
    from finder.pipeline import find_devices as _find

    resolution, devices = _find(
        analyte,
        extra_synonyms=extra_synonyms,
        resolve_urls=resolve_summary_urls,
    )

    return {
        "analyte": resolution.analyte_term,
        "synonyms_used": resolution.synonyms_used,
        "product_codes": [
            {
                "product_code": p.product_code,
                "device_name": p.device_name,
                "regulation_number": p.regulation_number,
                "device_class": p.device_class,
                "medical_specialty": p.medical_specialty,
            }
            for p in resolution.product_codes
        ],
        "heuristic_note": resolution.note,
        "devices": [
            {
                "k_number": d.k_number,
                "device_name": d.device_name,
                "applicant_name": d.applicant_name,
                "decision_date": str(d.decision_date) if d.decision_date else None,
                "product_code": d.product_code,
                "regulation_number": d.regulation_number,
                "device_class": d.device_class,
                "summary_url": d.summary_url,
            }
            for d in devices
        ],
        "total_devices": len(devices),
    }


# ---------------------------------------------------------------------------
# Tool: get_clearance
# ---------------------------------------------------------------------------

@mcp.tool(
    annotations=_READ_ONLY,
    description=(
        "Look up a single 510(k) clearance by K-number. "
        "Returns the openFDA record plus the index status of the Summary PDF "
        "(whether it has been fetched and chunked for Q&A)."
    ),
)
def get_clearance(k_number: str) -> dict:
    """
    k_number: the FDA K-number (e.g. 'K173653')
    """
    from finder.sources.openfda import get_510k_by_knumber
    from finder.sources.summaries import resolve_summary_url
    from finder.index.store import get_index_status, load_chunks

    rec = get_510k_by_knumber(k_number)
    if rec is None:
        return {"error": f"{k_number} not found in openFDA 510(k) database"}

    status = get_index_status(k_number)
    chunk_count = len(load_chunks(k_number)) if status == "ok" else 0
    summary_url = resolve_summary_url(k_number)

    return {
        "k_number": rec.get("k_number"),
        "device_name": rec.get("device_name"),
        "applicant_name": rec.get("applicant_name"),
        "decision_date": rec.get("decision_date"),
        "product_code": rec.get("product_code"),
        "device_class": rec.get("device_class"),
        "statement_or_summary": rec.get("statement_or_summary"),
        "decision_code": rec.get("decision_code"),
        "summary_url": summary_url,
        "index_status": status or "not_indexed",
        "indexed_chunk_count": chunk_count,
    }


# ---------------------------------------------------------------------------
# Tool: ask_summary
# ---------------------------------------------------------------------------

@mcp.tool(
    annotations=_READ_ONLY,
    description=(
        "Answer a question grounded in indexed 510(k) Summary content. "
        "Runs in keyword-only mode (no LLM generation) — returns the most "
        "relevant chunk verbatim with a citation to K-number + page + section. "
        "Returns not_found if the indexed summaries don't contain an answer. "
        "Scope with k_numbers to restrict retrieval."
    ),
)
def ask_summary(
    question: str,
    k_numbers: list[str] | None = None,
    product_codes: list[str] | None = None,
    top_k: int = 5,
) -> dict:
    """
    question: the question to answer (e.g. 'What LoD did K173653 report?')
    k_numbers: scope retrieval to these K-numbers
    product_codes: scope retrieval to these product codes
    top_k: number of chunks to retrieve (default 5)
    """
    from finder.qa import ask

    answer = ask(
        question,
        k_numbers=k_numbers,
        product_codes=product_codes,
        top_k=top_k,
        llm=None,  # keyword-only; caller can supply an LLM via the ask CLI
    )

    return {
        "question": answer.question,
        "answer": answer.answer or None,
        "not_found_reason": answer.not_found_reason,
        "citations": [
            {
                "k_number": c.k_number,
                "page": c.page,
                "section": c.section,
                "source_url": c.source_url,
            }
            for c in answer.citations
        ],
    }


# ---------------------------------------------------------------------------
# Tool: compare_performance
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tool: find_reference_labs
# ---------------------------------------------------------------------------

@mcp.tool(
    annotations=_READ_ONLY,
    description=(
        "Search public lab test directories for tests matching an analyte. "
        "Searches ARUP Laboratories and Mayo Clinic Laboratories (both confirmed "
        "allowable per robots.txt). "
        "Results are directory listings only — NOT FDA determinations. "
        "The data_source field on every result confirms this."
    ),
)
def find_reference_labs(
    analyte: str,
    labs: list[str] | None = None,
) -> dict:
    """
    analyte: analyte name to search (e.g. 'Group A Strep')
    labs: subset of ['arup', 'mayo'] to query (default: both)
    """
    from finder.sources.labs import find_reference_labs as _find_labs, ALLOWED_LABS

    try:
        results = _find_labs(analyte, labs=labs)
    except ValueError as e:
        return {"error": str(e), "allowed_labs": list(ALLOWED_LABS)}

    return {
        "analyte": analyte,
        "directory_lookup_note": "These are lab test directory listings, not FDA determinations.",
        "results": [
            {
                "lab_name": t.lab_name,
                "test_name": t.test_name,
                "test_code": t.test_code,
                "methodology": t.methodology,
                "specimen_type": t.specimen_type,
                "url": t.url,
                "snapshot_date": t.snapshot_date,
                "data_source": t.data_source,
            }
            for t in results
        ],
        "total": len(results),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
