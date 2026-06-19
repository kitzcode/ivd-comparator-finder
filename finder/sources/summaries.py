"""
Locate and fetch 510(k) Summary PDFs from accessdata.fda.gov.

URL patterns vary by clearance year. Not every K-number has a public Summary
(some have only a Statement; some have nothing). This module probes known
patterns per-record and returns the first URL that serves a PDF.

PDFs are cached to data/cache/pdf/<K-number>.pdf.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import httpx

_DEFAULT_CACHE = Path(__file__).parent.parent.parent / "data" / "cache" / "pdf"
try:
    _DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
    CACHE_DIR = _DEFAULT_CACHE
except OSError:
    # Read-only filesystem (e.g. Vercel serverless) — fall back to /tmp
    CACHE_DIR = Path("/tmp") / "ivd_pdf_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Also check for committed URL sidecar files in the project tree (readable even
# on read-only filesystems like Vercel where CACHE_DIR may point to /tmp).
_COMMITTED_URL_DIR = Path(__file__).parent.parent.parent / "data" / "cache" / "pdf"

_URL_PATTERNS = [
    "https://www.accessdata.fda.gov/cdrh_docs/pdf{yy}/{k}.pdf",
    "https://www.accessdata.fda.gov/cdrh_docs/reviews/{k}.pdf",
    "https://www.accessdata.fda.gov/cdrh_docs/pdf/{k}.pdf",
]

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; IVDFinder/1.0)"}
_PROBE_TIMEOUT = 6  # seconds per URL probe; 3 probes in parallel so wall-clock ≈ 6s max


def _year2(k_number: str) -> str:
    digits = re.sub(r"[^0-9]", "", k_number)
    return digits[:2] if len(digits) >= 2 else "00"


def _cache_pdf_path(k_number: str) -> Path:
    return CACHE_DIR / f"{k_number}.pdf"


def _cache_url_path(k_number: str) -> Path:
    return CACHE_DIR / f"{k_number}.url"


def _committed_url_path(k_number: str) -> Path:
    """Sidecar committed to git — readable on Vercel even when CACHE_DIR is /tmp."""
    return _COMMITTED_URL_DIR / f"{k_number}.url"


def resolve_summary_url(k_number: str) -> Optional[str]:
    """
    Probe known URL patterns in parallel and return the first that serves a PDF,
    or None if no public Summary is found.

    Checks committed sidecar files first (fast, no network), then /tmp cache,
    then probes all candidates concurrently (wall-clock = single slowest probe).
    """
    # 1. Check committed sidecar (readable on Vercel)
    committed = _committed_url_path(k_number)
    if committed.exists():
        val = committed.read_text().strip()
        return val if val != "NONE" else None

    # 2. Check writable cache sidecar
    url_file = _cache_url_path(k_number)
    if url_file.exists():
        val = url_file.read_text().strip()
        return val if val != "NONE" else None

    yy = _year2(k_number)
    candidates = [p.format(k=k_number, yy=yy) for p in _URL_PATTERNS]

    def _probe(url: str) -> Optional[str]:
        try:
            with httpx.Client(timeout=_PROBE_TIMEOUT, follow_redirects=True) as client:
                resp = client.get(url, headers={**_HEADERS, "Range": "bytes=0-3"})
                ct = resp.headers.get("content-type", "")
                if resp.status_code in (200, 206) and "pdf" in ct.lower():
                    return url
                if resp.status_code == 200 and resp.content[:4] == b"%PDF":
                    return url
        except httpx.RequestError:
            pass
        return None

    # Probe all candidates in parallel — wall-clock limited to slowest single probe.
    found: Optional[str] = None
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        futures = {pool.submit(_probe, url): url for url in candidates}
        # Honour URL_PATTERNS priority: prefer earlier patterns when multiple hit.
        results: dict[str, Optional[str]] = {}
        for future in as_completed(futures):
            url = futures[future]
            results[url] = future.result()
        for url in candidates:
            if results.get(url):
                found = url
                break

    try:
        url_file.write_text(found or "NONE")
    except OSError:
        pass  # read-only filesystem — skip caching

    return found


def fetch_summary_pdf(k_number: str) -> Optional[Path]:
    """
    Download the 510(k) Summary PDF for k_number to the cache dir.
    Returns the local path, or None if no public Summary exists.
    """
    local = _cache_pdf_path(k_number)
    if local.exists() and local.stat().st_size > 0:
        return local

    url = resolve_summary_url(k_number)
    if url is None:
        return None

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        try:
            resp = client.get(url, headers=_HEADERS)
            if resp.status_code != 200:
                return None
            local.write_bytes(resp.content)
            return local
        except httpx.RequestError:
            return None


def is_image_only_pdf(pdf_path: Path, sample_pages: int = 3) -> bool:
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:sample_pages]:
                if len((page.extract_text() or "").strip()) > 20:
                    return False
        return True
    except Exception:
        return False
