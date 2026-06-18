"""
M5: Reference-lab directory lookups.

These are public lab test directory pages (ARUP, Mayo Clinic Laboratories,
Quest Diagnostics, LabCorp), not FDA data. Results are labeled as directory
lookups and must never be presented as FDA determinations.

Access policy:
  - ARUP: robots.txt allows crawling; confirmed 2026-06-18.
  - Mayo Clinic Laboratories: robots.txt allows crawling; confirmed 2026-06-18.
  - Quest / LabCorp: only queried via their public search APIs or static pages.
  - No lab is accessed if its ToS or robots.txt prohibit automated access.
  - Results are cached to data/cache/labs/ with a snapshot date in the filename
    so runs are auditable and reproducible.

This module is intentionally conservative: it tries lightweight HTML scraping
with a clear User-Agent and falls back to returning empty results if the page
structure has changed or access is denied. Do not add new labs without
re-checking robots.txt and ToS.
"""

from __future__ import annotations

import json
import re
import time
from datetime import date
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel

_DEFAULT_LABS_CACHE = Path(__file__).parent.parent.parent / "data" / "cache" / "labs"
try:
    _DEFAULT_LABS_CACHE.mkdir(parents=True, exist_ok=True)
    CACHE_DIR = _DEFAULT_LABS_CACHE
except OSError:
    CACHE_DIR = Path("/tmp") / "ivd_labs_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_DATE = date.today().isoformat()
_HEADERS = {
    "User-Agent": "IVDComparatorFinder/1.0 (research tool; contact kitz.alexandra@gmail.com)"
}
_REQUEST_DELAY = 1.0  # seconds between requests to same host


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class LabTest(BaseModel):
    lab_name: str
    test_name: str
    test_code: Optional[str] = None
    methodology: Optional[str] = None
    specimen_type: Optional[str] = None
    url: Optional[str] = None
    snapshot_date: str = SNAPSHOT_DATE
    # Always label as a directory lookup, not an FDA determination
    data_source: str = "Lab test directory (not an FDA determination)"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(lab: str, analyte: str) -> str:
    slug = re.sub(r"[^a-z0-9]", "_", analyte.lower())[:40]
    return f"{lab}_{slug}_{SNAPSHOT_DATE}"


def _load_cache(key: str) -> Optional[list[dict]]:
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def _save_cache(key: str, data: list[dict]) -> None:
    try:
        (CACHE_DIR / f"{key}.json").write_text(json.dumps(data, indent=2))
    except OSError:
        pass  # read-only filesystem — skip caching


# ---------------------------------------------------------------------------
# ARUP Laboratories
# ---------------------------------------------------------------------------
# robots.txt (as of 2026-06-18): Disallow: /nothing — crawling allowed.
# Search endpoint: https://www.aruplab.com/Testing-Information/search?q=<term>

_ARUP_SEARCH = "https://www.aruplab.com/Testing-Information/search"


def _search_arup(analyte: str) -> list[LabTest]:
    key = _cache_key("arup", analyte)
    cached = _load_cache(key)
    if cached is not None:
        return [LabTest(**r) for r in cached]

    results: list[LabTest] = []
    try:
        with httpx.Client(timeout=20, follow_redirects=True) as client:
            resp = client.get(_ARUP_SEARCH, params={"q": analyte}, headers=_HEADERS)
            if resp.status_code != 200:
                return []
            html = resp.text

        # Extract test links and names from the search results page
        # Pattern: <a href="/Testing-Information/<id>">test name</a>
        for m in re.finditer(
            r'href="(/Testing-Information/([^"]+))"[^>]*>\s*([^<]{5,120})\s*</a>',
            html, re.I
        ):
            path, _, name = m.group(1), m.group(2), m.group(3).strip()
            name = re.sub(r"\s+", " ", name)
            if not _relevance_filter(analyte, name):
                continue
            url = f"https://www.aruplab.com{path}"

            # Fetch the individual test page for methodology/specimen
            method, specimen = _fetch_arup_test_detail(client, url)

            results.append(LabTest(
                lab_name="ARUP Laboratories",
                test_name=name,
                url=url,
                methodology=method,
                specimen_type=specimen,
            ))
            if len(results) >= 10:
                break

    except httpx.RequestError:
        pass

    _save_cache(key, [r.model_dump() for r in results])
    return results


def _fetch_arup_test_detail(client: httpx.Client, url: str) -> tuple[Optional[str], Optional[str]]:
    """Fetch individual ARUP test page and extract Methodology and Specimen."""
    try:
        time.sleep(_REQUEST_DELAY)
        resp = client.get(url, headers=_HEADERS)
        if resp.status_code != 200:
            return None, None
        html = resp.text
        method = _extract_field(html, "Methodology", r"([A-Za-z][^\n<]{3,120})")
        specimen = _extract_field(html, "Specimen", r"([A-Za-z][^\n<]{3,80})")
        return method, specimen
    except httpx.RequestError:
        return None, None


def _extract_field(html: str, label: str, value_re: str) -> Optional[str]:
    pat = re.compile(
        rf"{re.escape(label)}[^<]*</(?:dt|th|label|td)[^>]*>[^<]*<[^>]+>\s*{value_re}",
        re.I | re.S,
    )
    m = pat.search(html)
    if m:
        return m.group(1).strip()[:150]

    # Fallback: look for label: value on consecutive lines
    m2 = re.search(
        rf"{re.escape(label)}\s*[:\-–]\s*([A-Za-z][^\n<]{{3,120}})", html, re.I
    )
    return m2.group(1).strip()[:150] if m2 else None


# ---------------------------------------------------------------------------
# Mayo Clinic Laboratories
# ---------------------------------------------------------------------------
# robots.txt (as of 2026-06-18): Disallow entries cover /forms/ only.
# Search: https://www.mayocliniclabs.com/test-catalog/search?search_string=<term>

_MAYO_SEARCH = "https://www.mayocliniclabs.com/test-catalog/search"


def _search_mayo(analyte: str) -> list[LabTest]:
    key = _cache_key("mayo", analyte)
    cached = _load_cache(key)
    if cached is not None:
        return [LabTest(**r) for r in cached]

    results: list[LabTest] = []
    try:
        with httpx.Client(timeout=20, follow_redirects=True) as client:
            resp = client.get(_MAYO_SEARCH, params={"search_string": analyte}, headers=_HEADERS)
            if resp.status_code != 200:
                _save_cache(key, [])
                return []
            html = resp.text

        # Extract test cards: <a href="/test-catalog/Overview/<code>-...">name</a>
        for m in re.finditer(
            r'href="(/test-catalog/Overview/([^"]+))"[^>]*>\s*([^<]{5,120})\s*</a>',
            html, re.I,
        ):
            path, code_slug, name = m.group(1), m.group(2), m.group(3).strip()
            name = re.sub(r"\s+", " ", name)
            if not _relevance_filter(analyte, name):
                continue
            # Extract test code from the slug (e.g. "STREA-Overview-12345" → "12345")
            code_m = re.search(r"(\d{4,6})", code_slug)
            test_code = code_m.group(1) if code_m else None
            url = f"https://www.mayocliniclabs.com{path}"

            results.append(LabTest(
                lab_name="Mayo Clinic Laboratories",
                test_name=name,
                test_code=test_code,
                url=url,
            ))
            if len(results) >= 10:
                break

    except httpx.RequestError:
        pass

    _save_cache(key, [r.model_dump() for r in results])
    return results


# ---------------------------------------------------------------------------
# Relevance filter
# ---------------------------------------------------------------------------

def _relevance_filter(analyte: str, test_name: str) -> bool:
    """Return True if the test name appears relevant to the analyte."""
    analyte_tokens = set(re.findall(r"[a-z]+", analyte.lower()))
    name_lower = test_name.lower()
    # Require at least one meaningful token from the analyte to appear in the name
    meaningful = {t for t in analyte_tokens if len(t) > 3}
    return bool(meaningful & set(re.findall(r"[a-z]+", name_lower)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ALLOWED_LABS = {"arup", "mayo"}


def find_reference_labs(
    analyte: str,
    labs: Optional[list[str]] = None,
) -> list[LabTest]:
    """
    Search public lab test directories for tests matching analyte.

    labs: subset of ALLOWED_LABS to query (default: all allowed).
    Results are labeled as directory lookups, not FDA determinations.

    *** ToS note: only labs in ALLOWED_LABS are accessed. Adding a new lab
    requires re-checking its robots.txt and terms of service. ***
    """
    target_labs = [l.lower() for l in (labs or list(ALLOWED_LABS))]
    invalid = [l for l in target_labs if l not in ALLOWED_LABS]
    if invalid:
        raise ValueError(
            f"Labs not in allowlist: {invalid}. "
            f"Check robots.txt and ToS before adding: {ALLOWED_LABS}"
        )

    results: list[LabTest] = []
    for lab in target_labs:
        if lab == "arup":
            results.extend(_search_arup(analyte))
            time.sleep(_REQUEST_DELAY)
        elif lab == "mayo":
            results.extend(_search_mayo(analyte))
            time.sleep(_REQUEST_DELAY)

    return results


def format_lab_results(tests: list[LabTest]) -> str:
    if not tests:
        return "No reference lab tests found in directory search."

    lines = [
        "REFERENCE LAB DIRECTORY LOOKUP",
        "*** These are directory listings, not FDA determinations. ***\n",
    ]
    for t in tests:
        lines.append(f"  Lab       : {t.lab_name}")
        lines.append(f"  Test      : {t.test_name}")
        if t.test_code:
            lines.append(f"  Code      : {t.test_code}")
        if t.methodology:
            lines.append(f"  Method    : {t.methodology}")
        if t.specimen_type:
            lines.append(f"  Specimen  : {t.specimen_type}")
        if t.url:
            lines.append(f"  URL       : {t.url}")
        lines.append(f"  Snapshot  : {t.snapshot_date}")
        lines.append("")
    return "\n".join(lines)
