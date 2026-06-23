"""
openFDA Device MCP — the DATA layer.

Read-only, typed tools over openFDA device data. No RAG, no PDF parsing: this
server answers "what FDA-cleared devices and clearances exist" and nothing about
performance studies (that is the grounded_rag server's job).

Tools (readOnlyHint=True, destructiveHint=False):
  find_devices    analyte/assay term -> device table (K-numbers, product codes, predicates)
  get_clearance   single K-number -> full clearance record

Run:
  python -m mcp_servers.openfda_device
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
    "openfda-device",
    instructions=(
        "openFDA Device data layer. Maps analytes/assays to FDA-cleared devices and "
        "looks up individual clearances. Device discovery is heuristic (synonym text "
        "search); always surface the synonym set used. Read-only; public FDA data only."
    ),
)

_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False)


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


if __name__ == "__main__":
    mcp.run()
