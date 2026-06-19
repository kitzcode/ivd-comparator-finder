"""
FastAPI web application for the IVD Comparator Finder.

Endpoints:
  GET  /api/find          analyte -> device table
  POST /api/ingest        fetch + parse PDFs for given K-numbers
  GET  /api/clearance     single K-number -> openFDA record + index status
  GET  /api/ask           grounded Q&A (keyword mode)
  GET  /api/compare       structured performance extraction table
  GET  /api/labs          reference-lab directory lookup
  GET  /api/status        index manifest

The frontend is served from static/index.html.
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

app = FastAPI(
    title="IVD Comparator Finder",
    description="FDA 510(k) device lookup, PDF parsing, and grounded performance Q&A",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/find")
def api_find(
    analyte: str = Query(..., description="Analyte or assay name, e.g. 'Group A Strep'"),
    synonyms: str = Query("", description="Comma-separated extra synonyms"),
):
    from finder.pipeline import find_devices
    extra = [s.strip() for s in synonyms.split(",") if s.strip()] or None
    resolution, devices = find_devices(analyte, extra_synonyms=extra, resolve_urls=False)
    return {
        "analyte": resolution.analyte_term,
        "synonyms_used": resolution.synonyms_used,
        "heuristic_note": resolution.note,
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
        "devices": [
            {
                "k_number": d.k_number,
                "device_name": d.device_name,
                "applicant_name": d.applicant_name,
                "decision_date": str(d.decision_date) if d.decision_date else None,
                "product_code": d.product_code,
                "regulation_number": d.regulation_number,
                "device_class": d.device_class,
            }
            for d in devices
        ],
        "total": len(devices),
    }


@app.get("/api/clearance/{k_number}")
def api_clearance(k_number: str):
    from finder.sources.openfda import get_510k_by_knumber
    from finder.sources.summaries import resolve_summary_url
    from finder.index.store import get_index_status, load_chunks

    rec = get_510k_by_knumber(k_number.upper())
    if not rec:
        raise HTTPException(status_code=404, detail=f"{k_number} not found in openFDA")

    status = get_index_status(k_number.upper())
    chunk_count = len(load_chunks(k_number.upper())) if status == "ok" else 0
    summary_url = resolve_summary_url(k_number.upper())

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


@app.get("/api/ingest-stream")
def api_ingest_stream(k_numbers: str = Query(..., description="Comma-separated K-numbers")):
    """
    SSE endpoint — runs ingest in a thread and streams progress events in real
    time. Keeps the HTTP connection alive so Vercel doesn't terminate the
    function before the work completes.
    """
    import queue
    import threading
    from finder.models import Device
    from finder.sources.openfda import get_510k_by_knumber
    from finder.pipeline import ingest_summaries

    ks = [k.strip().upper() for k in k_numbers.split(",") if k.strip()]
    q: queue.Queue = queue.Queue()
    _DONE = object()

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def _stage(msg: str):
        lower = msg.lower()
        if "fetching" in lower:   return "downloading", 25
        if "no public" in lower:  return "no_summary",  30
        if "image-only" in lower: return "image_only",  60
        if "error" in lower:      return "error",        60
        if "ingested" in lower:   return "indexing",     95
        if "skipped" in lower:    return "done",        100
        return "parsing", 55

    def _worker():
        try:
            devices = []
            for k in ks:
                q.put(_sse("progress", {"k": k, "stage": "queued", "pct": 5, "msg": "Looking up device…"}))
                rec = get_510k_by_knumber(k)
                if not rec:
                    q.put(_sse("progress", {"k": k, "stage": "error", "pct": 0, "msg": f"{k} not found"}))
                    q.put(_sse("done", {"k": k, "status": "error"}))
                    continue
                devices.append(Device(
                    k_number=k,
                    device_name=rec.get("device_name", ""),
                    applicant_name=rec.get("applicant_name", ""),
                    product_code=rec.get("product_code", ""),
                ))

            def progress_cb(msg: str):
                parts = msg.split(":", 1)
                k = parts[0].strip()
                text = parts[1].strip() if len(parts) > 1 else msg
                stage, pct = _stage(text)
                q.put(_sse("progress", {"k": k, "stage": stage, "pct": pct, "msg": text}))

            results = ingest_summaries(devices, progress_cb=progress_cb, skip_already_indexed=True)
            for r in results:
                q.put(_sse("done", {"k": r.k_number, "status": r.status,
                                    "chunk_count": r.chunk_count, "note": r.note}))
            q.put(_sse("complete", {"ks": ks}))
        except Exception as exc:
            q.put(_sse("error", {"msg": str(exc)}))
        finally:
            q.put(_DONE)

    threading.Thread(target=_worker, daemon=True).start()

    def generate():
        while True:
            item = q.get()
            if item is _DONE:
                break
            yield item

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Keep the POST endpoint for backwards compat (used by old clients / CLI)
_ingest_progress: dict[str, str] = {}

@app.post("/api/ingest")
def api_ingest(k_numbers: list[str], background_tasks: BackgroundTasks):
    k_numbers = [k.upper() for k in k_numbers]
    for k in k_numbers:
        _ingest_progress[k] = "queued"

    def _run():
        from finder.models import Device
        from finder.sources.openfda import get_510k_by_knumber
        from finder.pipeline import ingest_summaries
        devices = []
        for k in k_numbers:
            rec = get_510k_by_knumber(k)
            if rec:
                devices.append(Device(k_number=k, device_name=rec.get("device_name",""),
                    applicant_name=rec.get("applicant_name",""), product_code=rec.get("product_code","")))
        def _cb(msg):
            _ingest_progress[msg.split(":")[0].strip()] = msg
        ingest_summaries(devices, progress_cb=_cb, skip_already_indexed=True)

    background_tasks.add_task(_run)
    return {"queued": k_numbers}


@app.get("/api/ask")
def api_ask(
    q: str = Query(..., description="Question to answer"),
    k_numbers: str = Query("", description="Comma-separated K-numbers to scope retrieval"),
    top_k: int = Query(5, ge=1, le=20),
):
    from finder.qa import ask

    kn = [k.strip().upper() for k in k_numbers.split(",") if k.strip()] or None
    answer = ask(q, k_numbers=kn, top_k=top_k)
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


@app.get("/api/compare")
def api_compare(
    k_numbers: str = Query(..., description="Comma-separated K-numbers"),
):
    from finder.extract import extract_performance
    from finder.sources.openfda import get_510k_by_knumber

    kn = [k.strip().upper() for k in k_numbers.split(",") if k.strip()]
    if not kn:
        raise HTTPException(status_code=400, detail="Provide at least one K-number")

    device_names: dict[str, str] = {}
    product_codes_map: dict[str, str] = {}
    for k in kn:
        rec = get_510k_by_knumber(k)
        if rec:
            device_names[k] = rec.get("device_name", "")
            product_codes_map[k] = rec.get("product_code", "")

    table = extract_performance(kn, device_names=device_names, product_codes=product_codes_map)

    def _pv(val):
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


@app.get("/api/labs")
def api_labs(
    analyte: str = Query(..., description="Analyte name"),
    labs: str = Query("", description="Comma-separated: arup,mayo (default: both)"),
):
    from finder.sources.labs import find_reference_labs, ALLOWED_LABS

    labs_list = [l.strip().lower() for l in labs.split(",") if l.strip()] or None
    try:
        results = find_reference_labs(analyte, labs=labs_list)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "analyte": analyte,
        "directory_lookup_note": "Directory listings only — not FDA determinations.",
        "results": [
            {
                "lab_name": t.lab_name,
                "test_name": t.test_name,
                "test_code": t.test_code,
                "methodology": t.methodology,
                "specimen_type": t.specimen_type,
                "url": t.url,
                "snapshot_date": t.snapshot_date,
            }
            for t in results
        ],
        "total": len(results),
    }


@app.get("/api/status")
def api_status():
    from finder.index.store import list_indexed, load_chunks

    manifest = list_indexed()
    return {
        "indexed": {
            k: {"status": v, "chunk_count": len(load_chunks(k)) if v == "ok" else 0}
            for k, v in manifest.items()
        },
        "ingest_progress": dict(_ingest_progress),
    }


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse("static/index.html")


@app.get("/{path:path}", include_in_schema=False)
def catch_all(path: str):
    return FileResponse("static/index.html")
