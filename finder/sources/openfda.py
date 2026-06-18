"""
openFDA client for device classification and 510(k) endpoints.

All results are cached to disk at data/cache/ with a snapshot date suffix.
Cache files are JSON; use them for deterministic replay in tests.

Rate limits (unauthenticated): ~240 req/min. With API key: ~240,000 req/day.
Set OPENFDA_API_KEY env var to raise limits.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE = "https://api.fda.gov"
CLASSIFICATION_EP = f"{BASE}/device/classification.json"
FIVEK_EP = f"{BASE}/device/510k.json"

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("OPENFDA_API_KEY", "")
REQUEST_DELAY = 0.3  # seconds between requests when not using an API key


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------

def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.json"


def _load_cache(name: str) -> Optional[dict]:
    p = _cache_path(name)
    if p.exists():
        return json.loads(p.read_text())
    return None


def _save_cache(name: str, data: dict) -> None:
    _cache_path(name).write_text(json.dumps(data, indent=2))


def _get(url: str, params: dict[str, Any], cache_key: str) -> dict:
    """Fetch URL with params, returning parsed JSON. Uses on-disk cache."""
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    if API_KEY:
        params = {**params, "api_key": API_KEY}

    with httpx.Client(timeout=30) as client:
        resp = client.get(url, params=params)
        # openFDA returns 404 when a search matches zero results — treat as empty
        if resp.status_code == 404:
            data: dict = {"results": [], "meta": {"results": {"total": 0}}}
            _save_cache(cache_key, data)
            return data
        resp.raise_for_status()
        data = resp.json()

    _save_cache(cache_key, data)

    if not API_KEY:
        time.sleep(REQUEST_DELAY)

    return data


def _get_all_pages(
    url: str,
    base_params: dict[str, Any],
    cache_key_prefix: str,
    limit: int = 100,
    max_results: int = 1000,
) -> list[dict]:
    """Paginate through all results, caching each page."""
    results: list[dict] = []
    skip = 0

    while skip < max_results:
        page_key = f"{cache_key_prefix}_skip{skip}"
        params = {**base_params, "limit": limit, "skip": skip}
        data = _get(url, params, page_key)

        hits = data.get("results", [])
        results.extend(hits)

        meta = data.get("meta", {}).get("results", {})
        total = meta.get("total", 0)

        if skip + limit >= total or not hits:
            break
        skip += limit

    return results


# ---------------------------------------------------------------------------
# Device Classification endpoint
# ---------------------------------------------------------------------------

def get_classification_by_product_code(product_code: str) -> list[dict]:
    """Return classification records for a product code."""
    cache_key = f"cls_pc_{product_code}"
    data = _get(
        CLASSIFICATION_EP,
        {"search": f'product_code:"{product_code}"', "limit": 10},
        cache_key,
    )
    return data.get("results", [])


def search_classification_by_term(term: str, limit: int = 100) -> list[dict]:
    """
    Text-search device_name and definition fields for term (OR across both fields).
    Returns raw classification records.
    """
    safe_term = term.replace('"', '\\"')
    # Use explicit OR; openFDA treats space-separated terms as OR within a field
    search = f'(device_name:"{safe_term}"+OR+definition:"{safe_term}")'
    cache_key = f"cls_term_{term.lower().replace(' ', '_')[:60]}"
    data = _get(CLASSIFICATION_EP, {"search": search, "limit": limit}, cache_key)
    return data.get("results", [])


# ---------------------------------------------------------------------------
# 510(k) endpoint
# ---------------------------------------------------------------------------

def get_510k_by_product_code(product_code: str, max_results: int = 500) -> list[dict]:
    """Return all 510(k) clearances for a product code."""
    cache_key_prefix = f"fivek_pc_{product_code}"
    return _get_all_pages(
        FIVEK_EP,
        {"search": f'product_code:"{product_code}"'},
        cache_key_prefix,
        max_results=max_results,
    )


def search_510k_by_term(term: str, limit: int = 100) -> list[dict]:
    """
    Text-search 510(k) device_name for term.
    Returns raw 510(k) records.
    """
    safe_term = term.replace('"', '\\"')
    cache_key = f"fivek_term_{term.lower().replace(' ', '_')[:60]}"
    data = _get(
        FIVEK_EP,
        {"search": f'device_name:"{safe_term}"', "limit": limit},
        cache_key,
    )
    return data.get("results", [])


def get_510k_by_knumber(k_number: str) -> Optional[dict]:
    """Return a single 510(k) record by K-number."""
    cache_key = f"fivek_k_{k_number}"
    data = _get(
        FIVEK_EP,
        {"search": f'k_number:"{k_number}"', "limit": 1},
        cache_key,
    )
    results = data.get("results", [])
    return results[0] if results else None


def clear_cache_for_term(term: str) -> None:
    """Remove cached files matching a term (for re-fetching)."""
    slug = term.lower().replace(" ", "_")[:60]
    for p in CACHE_DIR.glob(f"*{slug}*"):
        p.unlink()
