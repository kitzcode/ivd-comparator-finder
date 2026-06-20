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
from .sources.openfda import get_510k_by_product_code, get_pma_by_product_code
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
    """Normalize a 510(k)/De Novo record from /device/510k.json."""
    k = rec.get("k_number", "")
    # De Novo grants live in the 510(k) dataset with DEN-prefixed numbers
    # (decision_code DENG). Everything else from this endpoint is a 510(k).
    is_denovo = k.upper().startswith("DEN") or rec.get("decision_code", "").upper().startswith("DEN")
    return Device(
        k_number=k,
        device_name=rec.get("device_name", ""),
        applicant_name=rec.get("applicant_name", ""),
        decision_date=_parse_date(rec.get("decision_date_as_string") or rec.get("date_received")),
        product_code=product_code,
        submission_type="De Novo" if is_denovo else "510(k)",
        regulation_number=regulation_number,
        device_class=rec.get("device_class"),
        predicate_k_number=rec.get("traditional_501k_flag"),
        predicate_device_name=None,
    )


def _normalize_pma(rec: dict, product_code: str, regulation_number: Optional[str]) -> Device:
    """Normalize a PMA record from /device/pma.json (different field names)."""
    return Device(
        k_number=rec.get("pma_number", ""),
        device_name=rec.get("trade_name") or rec.get("generic_name", ""),
        applicant_name=rec.get("applicant", ""),
        decision_date=_parse_date(rec.get("decision_date") or rec.get("date_received")),
        product_code=product_code,
        submission_type="PMA",
        regulation_number=regulation_number,
        device_class="3",  # PMA == Class III
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
    seen_pma: set[str] = set()

    # Fetch 510(k) (incl. De Novo) and PMA device lists for every product code
    # in parallel to keep cold-cache latency low.
    def _fetch_5k(pc_info):
        return "5k", pc_info, get_510k_by_product_code(pc_info.product_code, max_results=200)

    def _fetch_pma(pc_info):
        return "pma", pc_info, get_pma_by_product_code(pc_info.product_code, max_results=200)

    pcs = resolution.product_codes
    max_workers = min(16, (len(pcs) * 2) or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for pc in pcs:
            futures.append(pool.submit(_fetch_5k, pc))
            futures.append(pool.submit(_fetch_pma, pc))
        for future in as_completed(futures):
            try:
                kind, pc_info, records = future.result()
            except Exception:
                continue
            for rec in records:
                if kind == "5k":
                    k = rec.get("k_number", "")
                    if not k or k in seen_k:
                        continue
                    seen_k.add(k)
                    dev = _normalize_device(rec, pc_info.product_code, pc_info.regulation_number)
                    if resolve_urls and dev.k_number:
                        dev = dev.model_copy(update={"summary_url": resolve_summary_url(dev.k_number)})
                    devices.append(dev)
                else:  # PMA — dedupe supplements down to one row per PMA number
                    pma = rec.get("pma_number", "")
                    if not pma or pma in seen_pma:
                        continue
                    seen_pma.add(pma)
                    devices.append(_normalize_pma(rec, pc_info.product_code, pc_info.regulation_number))

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
