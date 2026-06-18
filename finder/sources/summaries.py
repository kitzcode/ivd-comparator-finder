"""
Locate and fetch 510(k) Summary PDFs from accessdata.fda.gov.

URL patterns vary by clearance year. Not every K-number has a public Summary
(some have only a Statement; some have nothing). This module probes known
patterns per-record and returns the first URL that serves a PDF.

PDFs are cached to data/cache/pdf/<K-number>.pdf.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import httpx

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache" / "pdf"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Probe order matters: more-specific patterns first.
# {yy} = two-digit year extracted from the K-number.
_URL_PATTERNS = [
    "https://www.accessdata.fda.gov/cdrh_docs/pdf{yy}/{k}.pdf",
    "https://www.accessdata.fda.gov/cdrh_docs/reviews/{k}.pdf",
    "https://www.accessdata.fda.gov/cdrh_docs/pdf/{k}.pdf",
]


def _year2(k_number: str) -> str:
    digits = re.sub(r"[^0-9]", "", k_number)
    return digits[:2] if len(digits) >= 2 else "00"


def _cache_pdf_path(k_number: str) -> Path:
    return CACHE_DIR / f"{k_number}.pdf"


def _cache_url_path(k_number: str) -> Path:
    """Tiny sidecar that stores the resolved URL so we don't re-probe."""
    return CACHE_DIR / f"{k_number}.url"


def resolve_summary_url(k_number: str) -> Optional[str]:
    """
    Probe known URL patterns and return the first that serves a PDF,
    or None if no public Summary is found.

    Results are cached in a small sidecar file so repeated calls are free.
    """
    url_file = _cache_url_path(k_number)
    if url_file.exists():
        val = url_file.read_text().strip()
        return val if val != "NONE" else None

    yy = _year2(k_number)
    candidates = [p.format(k=k_number, yy=yy) for p in _URL_PATTERNS]

    found: Optional[str] = None
    headers = {"User-Agent": "Mozilla/5.0 (compatible; IVDFinder/1.0)"}
    with httpx.Client(timeout=20, follow_redirects=True) as client:
        for url in candidates:
            try:
                # accessdata.fda.gov returns 404 on HEAD; use GET with Range
                # to avoid downloading the entire PDF just to probe existence.
                resp = client.get(url, headers={**headers, "Range": "bytes=0-3"})
                ct = resp.headers.get("content-type", "")
                if resp.status_code in (200, 206) and "pdf" in ct.lower():
                    found = url
                    break
                # Some servers ignore Range and return 200 with content anyway
                if resp.status_code == 200 and resp.content[:4] == b"%PDF":
                    found = url
                    break
            except httpx.RequestError:
                continue

    url_file.write_text(found or "NONE")
    return found


def fetch_summary_pdf(k_number: str) -> Optional[Path]:
    """
    Download the 510(k) Summary PDF for k_number to the cache dir.
    Returns the local path, or None if no public Summary exists.

    Uses the on-disk cache; will not re-download an already-cached PDF.
    """
    local = _cache_pdf_path(k_number)
    if local.exists() and local.stat().st_size > 0:
        return local

    url = resolve_summary_url(k_number)
    if url is None:
        return None

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        try:
            resp = client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; IVDFinder/1.0)"})
            if resp.status_code != 200:
                return None
            local.write_bytes(resp.content)
            return local
        except httpx.RequestError:
            return None


def is_image_only_pdf(pdf_path: Path, sample_pages: int = 3) -> bool:
    """
    Heuristic: if the first few pages extract < 20 chars of text each,
    the PDF is likely a scanned image and needs OCR.
    """
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages = pdf.pages[:sample_pages]
            for page in pages:
                text = page.extract_text() or ""
                if len(text.strip()) > 20:
                    return False
        return True
    except Exception:
        return False
