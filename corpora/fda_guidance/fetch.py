"""
Fetch FDA guidance PDFs from fda.gov/media/{media_id}/download.

Same technique as the 510(k) Summary prober: GET with a tiny Range header to
confirm a PDF is served before downloading (HEAD is unreliable on FDA hosts).
PDFs are cached to data/cache/guidance_pdf/, falling back to /tmp on read-only
filesystems (e.g. serverless).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import httpx

_DEFAULT_CACHE = Path(__file__).parent.parent.parent / "data" / "cache" / "guidance_pdf"
try:
    _DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
    CACHE_DIR = _DEFAULT_CACHE
except OSError:
    CACHE_DIR = Path("/tmp") / "ivd_guidance_pdf"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; IVDFinder/1.0)"}
_PROBE_TIMEOUT = 6


def media_url(media_id: str) -> str:
    return f"https://www.fda.gov/media/{media_id}/download"


def _pdf_path(media_id: str) -> Path:
    return CACHE_DIR / f"{media_id}.pdf"


def is_pdf_available(media_id: str) -> bool:
    """Probe whether the media id serves a PDF (Range GET, no full download)."""
    try:
        with httpx.Client(timeout=_PROBE_TIMEOUT, follow_redirects=True) as client:
            resp = client.get(media_url(media_id), headers={**_HEADERS, "Range": "bytes=0-3"})
            ct = resp.headers.get("content-type", "")
            if resp.status_code in (200, 206) and "pdf" in ct.lower():
                return True
            return resp.status_code == 200 and resp.content[:4] == b"%PDF"
    except httpx.RequestError:
        return False


def fetch_guidance_pdf(media_id: str) -> Optional[Path]:
    """Download the guidance PDF to the cache. Returns the local path or None."""
    cached = _pdf_path(media_id)
    if cached.exists() and cached.stat().st_size > 0:
        return cached

    try:
        with httpx.Client(timeout=90, follow_redirects=True) as client:
            resp = client.get(media_url(media_id), headers=_HEADERS)
            if resp.status_code != 200 or resp.content[:4] != b"%PDF":
                return None
            try:
                cached.write_bytes(resp.content)
                return cached
            except OSError:
                tmp = Path("/tmp") / "ivd_guidance_pdf"
                tmp.mkdir(parents=True, exist_ok=True)
                p = tmp / f"{media_id}.pdf"
                p.write_bytes(resp.content)
                return p
    except httpx.RequestError:
        return None
