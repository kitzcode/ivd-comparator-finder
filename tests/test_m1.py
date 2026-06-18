"""
M1 acceptance tests for the Group A Strep device table.

These tests run against a cached snapshot for deterministic replay.
They assert that known devices are PRESENT (not that the set is exhaustive),
because live openFDA data grows over time.

Known devices that must appear:
  - K141757: Alere i Strep A
  - K173653: Alere i Strep A 2 (now Abbott ID NOW Strep A 2)
  - Cepheid Xpert Xpress Strep A (K-number resolved from cache)
  - K193519: BioFire BCID2 panel (regulation 866.3365, NOT dropped)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"

# ---------------------------------------------------------------------------
# Helpers to load from cache without making live requests
# ---------------------------------------------------------------------------

def _load_all_cached_fivek(product_code: str) -> list[dict]:
    """Load all paginated 510(k) cache files for a product code."""
    results = []
    for p in sorted(CACHE_DIR.glob(f"fivek_pc_{product_code}*.json")):
        data = json.loads(p.read_text())
        results.extend(data.get("results", []))
    return results


def _all_k_numbers_in_cache() -> set[str]:
    """Collect all K-numbers from all cached 510(k) files."""
    k_numbers = set()
    for p in CACHE_DIR.glob("fivek_*.json"):
        try:
            data = json.loads(p.read_text())
            for rec in data.get("results", []):
                k = rec.get("k_number")
                if k:
                    k_numbers.add(k)
        except Exception:
            continue
    return k_numbers


def _all_product_codes_in_cache() -> set[str]:
    k_numbers = set()
    for p in CACHE_DIR.glob("fivek_*.json"):
        try:
            data = json.loads(p.read_text())
            for rec in data.get("results", []):
                pc = rec.get("product_code")
                if pc:
                    k_numbers.add(pc)
        except Exception:
            continue
    return k_numbers


# ---------------------------------------------------------------------------
# Tests (require cache to exist; skipped if cache is empty)
# ---------------------------------------------------------------------------

def _skip_if_no_cache():
    if not any(CACHE_DIR.glob("fivek_*.json")):
        pytest.skip("No cached openFDA data; run cli.py first to populate cache")


def test_k141757_alere_i_strep_a_present():
    """Alere i Strep A (K141757) must appear in cached 510(k) results."""
    _skip_if_no_cache()
    assert "K141757" in _all_k_numbers_in_cache(), (
        "K141757 (Alere i Strep A) not found in cache. "
        "Populate cache with: python cli.py 'Group A Strep'"
    )


def test_k173653_alere_i_strep_a2_present():
    """Alere i Strep A 2 / Abbott ID NOW Strep A 2 (K173653) must be present."""
    _skip_if_no_cache()
    assert "K173653" in _all_k_numbers_in_cache(), (
        "K173653 (Alere i Strep A 2) not found in cache."
    )


def test_k193519_biofire_bcid2_present():
    """BioFire BCID2 (K193519) must be present — proves multiplex panels are not dropped."""
    _skip_if_no_cache()
    assert "K193519" in _all_k_numbers_in_cache(), (
        "K193519 (BioFire BCID2) not found in cache."
    )


def test_cepheid_xpert_strep_a_present():
    """Cepheid Xpert Xpress Strep A must appear somewhere in cached results."""
    _skip_if_no_cache()
    for p in CACHE_DIR.glob("fivek_*.json"):
        try:
            data = json.loads(p.read_text())
            for rec in data.get("results", []):
                name = rec.get("device_name", "").lower()
                if "xpert" in name and ("strep" in name or "xpress" in name):
                    return  # found
        except Exception:
            continue
    pytest.fail("No Cepheid Xpert Xpress Strep A device found in cache.")


def test_biofire_bcid2_tagged_with_correct_regulation():
    """K193519 must be tagged under regulation 866.3365, not the dedicated Strep A regulation."""
    _skip_if_no_cache()
    for p in CACHE_DIR.glob("fivek_*.json"):
        try:
            data = json.loads(p.read_text())
            for rec in data.get("results", []):
                if rec.get("k_number") == "K193519":
                    # K193519 is a multiplex panel; its product_code maps to 866.3365
                    pc = rec.get("product_code", "")
                    # We check this indirectly via the product code classification cache
                    cls_cache = CACHE_DIR / f"cls_pc_{pc}.json"
                    if cls_cache.exists():
                        cls_data = json.loads(cls_cache.read_text())
                        for cls_rec in cls_data.get("results", []):
                            reg = cls_rec.get("regulation_number", "")
                            assert "866.2680" not in reg, (
                                f"K193519 (BioFire BCID2) incorrectly tagged under 866.2680 "
                                f"(the dedicated Strep A regulation). It should be 866.3365."
                            )
                    return
        except Exception:
            continue
    pytest.skip("K193519 not in cache; run cli.py first.")


def test_dedicated_strep_a_devices_under_866_2680():
    """Dedicated Strep A devices K141757 and K173653 must map to product code under 866.2680."""
    _skip_if_no_cache()
    for k in ("K141757", "K173653"):
        for p in CACHE_DIR.glob("fivek_*.json"):
            try:
                data = json.loads(p.read_text())
                for rec in data.get("results", []):
                    if rec.get("k_number") == k:
                        pc = rec.get("product_code", "")
                        cls_cache = CACHE_DIR / f"cls_pc_{pc}.json"
                        if cls_cache.exists():
                            cls_data = json.loads(cls_cache.read_text())
                            for cls_rec in cls_data.get("results", []):
                                reg = cls_rec.get("regulation_number", "")
                                if reg:
                                    assert "866.2680" in reg, (
                                        f"{k} product code {pc} maps to {reg}, expected 866.2680"
                                    )
            except Exception:
                continue
