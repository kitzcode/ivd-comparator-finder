"""
Pipeline orchestration for all milestones.

v1: analyte term -> list of cleared Device objects with metadata.
v2: ingest_summaries() -> fetch + parse + chunk + store 510(k) Summary PDFs.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Callable, Optional

from .analyte import resolve_analyte
from .models import AnalyteResolution, Device, SummaryChunk
from .sources.openfda import get_510k_by_product_code
from .sources.summaries import fetch_summary_pdf, resolve_summary_url


def _parse_date(raw: Optional[str]) -> Optional[date]:
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        try:
            return date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))
        except Exception:
            return None


def _normalize_device(rec: dict, product_code: str, regulation_number: Optional[str]) -> Device:
    return Device(
        k_number=rec.get("k_number", ""),
        device_name=rec.get("device_name", ""),
        applicant_name=rec.get("applicant_name", ""),
        decision_date=_parse_date(rec.get("decision_date_as_string") or rec.get("date_received")),
        product_code=product_code,
        regulation_number=regulation_number,
        device_class=rec.get("device_class"),
        predicate_k_number=rec.get("traditional_501k_flag"),
        predicate_device_name=None,
    )


# ---------------------------------------------------------------------------
# v1 entry point
# ---------------------------------------------------------------------------

def find_devices(
    analyte_term: str,
    extra_synonyms: Optional[list[str]] = None,
    resolve_urls: bool = False,
    medical_specialty: Optional[str] = None,
) -> tuple[AnalyteResolution, list[Device]]:
    """
    Analyte term -> resolved product codes -> full device table.
    Devices are sorted by decision_date descending (most recent first).
    """
    resolution = resolve_analyte(analyte_term, extra_synonyms, medical_specialty=medical_specialty)

    devices: list[Device] = []
    seen_k: set[str] = set()

    # Fetch 510(k) device lists for all product codes in parallel to reduce
    # wall-clock time when the cache is cold (e.g. first search of a new analyte).
    def _fetch(pc_info):
        return pc_info, get_510k_by_product_code(pc_info.product_code, max_results=200)

    max_workers = min(16, len(resolution.product_codes) or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch, pc): pc for pc in resolution.product_codes}
        for future in as_completed(futures):
            try:
                pc_info, records = future.result()
            except Exception:
                continue
            for rec in records:
                k = rec.get("k_number", "")
                if k in seen_k:
                    continue
                seen_k.add(k)
                dev = _normalize_device(rec, pc_info.product_code, pc_info.regulation_number)
                if resolve_urls and dev.k_number:
                    dev = dev.model_copy(update={"summary_url": resolve_summary_url(dev.k_number)})
                devices.append(dev)

    devices.sort(key=lambda d: d.decision_date or date.min, reverse=True)
    return resolution, devices


# ---------------------------------------------------------------------------
# v2 entry point: fetch + parse + store Summary PDFs
# ---------------------------------------------------------------------------

class IngestResult:
    """Summary of a single-K ingest attempt."""
    def __init__(self, k_number: str, status: str, chunk_count: int = 0, note: str = ""):
        self.k_number = k_number
        self.status = status      # 'ok' | 'no_summary' | 'image_only' | 'error'
        self.chunk_count = chunk_count
        self.note = note

    def __repr__(self) -> str:
        return f"IngestResult({self.k_number}, {self.status}, {self.chunk_count} chunks)"


def ingest_summaries(
    devices: list[Device],
    progress_cb: Optional[Callable[[str], None]] = None,
    skip_already_indexed: bool = True,
) -> list[IngestResult]:
    """
    For each device, fetch its 510(k) Summary PDF, parse it into sections,
    chunk it, and store to the index.

    progress_cb is called with a status string after each device.
    skip_already_indexed=True (default) avoids re-fetching cached PDFs.

    Returns a list of IngestResult objects (one per device).
    """
    from .parse.pdf import extract_pdf
    from .parse.sections import chunk_pdf
    from .index.store import store_chunks, is_indexed, get_index_status

    results: list[IngestResult] = []

    for dev in devices:
        k = dev.k_number
        if not k:
            continue

        if skip_already_indexed and is_indexed(k):
            status = get_index_status(k) or "ok"
            existing = _count_stored_chunks(k)
            r = IngestResult(k, status, existing, "already indexed")
            results.append(r)
            if progress_cb:
                progress_cb(f"{k}: skipped ({status}, {existing} chunks)")
            continue

        if progress_cb:
            progress_cb(f"{k}: fetching PDF …")

        pdf_path = fetch_summary_pdf(k)

        if pdf_path is None:
            store_chunks(k, [], status="no_summary")
            results.append(IngestResult(k, "no_summary", 0, "no public Summary PDF found"))
            if progress_cb:
                progress_cb(f"{k}: no public Summary PDF")
            continue

        try:
            pdf_content = extract_pdf(pdf_path, k)
        except Exception as exc:
            store_chunks(k, [], status="error")
            results.append(IngestResult(k, "error", 0, str(exc)))
            if progress_cb:
                progress_cb(f"{k}: extraction error — {exc}")
            continue

        if pdf_content.is_image_only:
            store_chunks(k, [], status="image_only")
            results.append(IngestResult(k, "image_only", 0, "PDF is image-only; OCR needed"))
            if progress_cb:
                progress_cb(f"{k}: image-only PDF (OCR required)")
            continue

        source_url = resolve_summary_url(k) or str(pdf_path)
        chunks = chunk_pdf(pdf_content, product_code=dev.product_code, source_url=source_url)
        store_chunks(k, chunks, status="ok")

        results.append(IngestResult(k, "ok", len(chunks)))
        if progress_cb:
            progress_cb(f"{k}: ingested {len(chunks)} chunks from {pdf_content.page_count} pages")

    return results


def _count_stored_chunks(k_number: str) -> int:
    from .index.store import load_chunks
    return len(load_chunks(k_number))
