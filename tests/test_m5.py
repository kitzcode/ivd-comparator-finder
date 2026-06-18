"""
M5 tests: reference-lab directory lookups.

Unit tests mock HTTP. Live tests are gated behind a --run-live flag to avoid
hitting external sites in CI (and respecting the ToS-gated constraint).

Run live: pytest tests/test_m5.py --run-live
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def pytest_addoption(parser):
    try:
        parser.addoption("--run-live", action="store_true", default=False)
    except ValueError:
        pass  # already added by another conftest


@pytest.fixture
def run_live(request):
    return request.config.getoption("--run-live", default=False)


# ---------------------------------------------------------------------------
# LabTest model
# ---------------------------------------------------------------------------

def test_lab_test_model():
    from finder.sources.labs import LabTest
    t = LabTest(lab_name="ARUP Laboratories", test_name="Strep A, Rapid")
    assert t.data_source == "Lab test directory (not an FDA determination)"
    assert t.lab_name == "ARUP Laboratories"


def test_lab_test_snapshot_date_is_iso():
    from finder.sources.labs import LabTest
    import re
    t = LabTest(lab_name="ARUP Laboratories", test_name="Test")
    assert re.match(r"\d{4}-\d{2}-\d{2}", t.snapshot_date)


# ---------------------------------------------------------------------------
# Allowlist enforcement
# ---------------------------------------------------------------------------

def test_invalid_lab_raises():
    from finder.sources.labs import find_reference_labs
    with pytest.raises(ValueError, match="not in allowlist"):
        find_reference_labs("Group A Strep", labs=["labcorp"])


def test_allowlist_covers_arup_and_mayo():
    from finder.sources.labs import ALLOWED_LABS
    assert "arup" in ALLOWED_LABS
    assert "mayo" in ALLOWED_LABS


# ---------------------------------------------------------------------------
# Relevance filter
# ---------------------------------------------------------------------------

def test_relevance_filter_passes_matching_name():
    from finder.sources.labs import _relevance_filter
    assert _relevance_filter("Group A Strep", "Streptococcus Group A, Rapid") is True


def test_relevance_filter_rejects_unrelated():
    from finder.sources.labs import _relevance_filter
    assert _relevance_filter("Group A Strep", "Glucose Tolerance Test") is False


# ---------------------------------------------------------------------------
# Cache read/write
# ---------------------------------------------------------------------------

def test_cache_roundtrip(tmp_path, monkeypatch):
    from finder.sources import labs as l
    monkeypatch.setattr(l, "CACHE_DIR", tmp_path)
    from finder.sources.labs import LabTest, _cache_key, _load_cache, _save_cache
    key = _cache_key("arup", "Group A Strep")
    data = [LabTest(lab_name="ARUP Laboratories", test_name="Strep A").model_dump()]
    _save_cache(key, data)
    loaded = _load_cache(key)
    assert loaded is not None
    assert loaded[0]["test_name"] == "Strep A"


# ---------------------------------------------------------------------------
# Mocked HTTP tests
# ---------------------------------------------------------------------------

_ARUP_HTML_FIXTURE = """
<html><body>
<a href="/Testing-Information/group-a-strep-12345">Streptococcus Group A, Rapid, Throat</a>
<a href="/Testing-Information/glucose-99999">Glucose Test (unrelated)</a>
</body></html>
"""

_ARUP_DETAIL_HTML = """
<html><body>
<dt>Methodology</dt><dd>Nucleic Acid Amplification (NAAT)</dd>
<dt>Specimen</dt><dd>Throat Swab</dd>
</body></html>
"""


def test_search_arup_mocked(tmp_path, monkeypatch):
    from finder.sources import labs as l
    monkeypatch.setattr(l, "CACHE_DIR", tmp_path)
    # Clear any cached URL sidecar
    import httpx
    from unittest.mock import patch, MagicMock

    search_resp = MagicMock()
    search_resp.status_code = 200
    search_resp.text = _ARUP_HTML_FIXTURE

    detail_resp = MagicMock()
    detail_resp.status_code = 200
    detail_resp.text = _ARUP_DETAIL_HTML

    def fake_get(url, **kwargs):
        if "search" in url:
            return search_resp
        return detail_resp

    with patch.object(httpx.Client, "get", side_effect=fake_get):
        results = l._search_arup("Group A Strep")

    assert len(results) >= 1
    assert results[0].lab_name == "ARUP Laboratories"
    assert "Strep" in results[0].test_name
    # Methodology and specimen from detail page
    assert results[0].methodology is not None or results[0].specimen_type is not None


_MAYO_HTML_FIXTURE = """
<html><body>
<a href="/test-catalog/Overview/STREA-Overview-12345">Streptococcus pyogenes (Group A), Throat Culture</a>
<a href="/test-catalog/Overview/GLUC-Overview-99999">Glucose, Serum (unrelated)</a>
</body></html>
"""


def test_search_mayo_mocked(tmp_path, monkeypatch):
    from finder.sources import labs as l
    monkeypatch.setattr(l, "CACHE_DIR", tmp_path)
    import httpx

    resp = MagicMock()
    resp.status_code = 200
    resp.text = _MAYO_HTML_FIXTURE

    with patch.object(httpx.Client, "get", return_value=resp):
        results = l._search_mayo("Group A Strep")

    assert len(results) >= 1
    assert results[0].lab_name == "Mayo Clinic Laboratories"
    assert "Strep" in results[0].test_name or "pyogenes" in results[0].test_name


def test_find_reference_labs_mocked(tmp_path, monkeypatch):
    from finder.sources import labs as l
    monkeypatch.setattr(l, "CACHE_DIR", tmp_path)
    import httpx

    search_resp = MagicMock(status_code=200, text=_ARUP_HTML_FIXTURE)
    detail_resp = MagicMock(status_code=200, text=_ARUP_DETAIL_HTML)
    mayo_resp = MagicMock(status_code=200, text=_MAYO_HTML_FIXTURE)

    call_count = [0]
    def fake_get(url, **kwargs):
        call_count[0] += 1
        if "aruplab.com/Testing-Information/search" in url:
            return search_resp
        if "mayocliniclabs" in url:
            return mayo_resp
        return detail_resp

    with patch.object(httpx.Client, "get", side_effect=fake_get):
        results = l.find_reference_labs("Group A Strep", labs=["arup", "mayo"])

    assert len(results) >= 1
    lab_names = {r.lab_name for r in results}
    assert "ARUP Laboratories" in lab_names or "Mayo Clinic Laboratories" in lab_names


def test_format_lab_results_labels_as_directory_lookup(tmp_path, monkeypatch):
    from finder.sources import labs as l
    from finder.sources.labs import LabTest, format_lab_results
    monkeypatch.setattr(l, "CACHE_DIR", tmp_path)
    tests = [LabTest(lab_name="ARUP Laboratories", test_name="Strep A, Rapid")]
    output = format_lab_results(tests)
    assert "DIRECTORY LOOKUP" in output
    assert "not fda" in output.lower()


def test_format_lab_results_empty():
    from finder.sources.labs import format_lab_results
    output = format_lab_results([])
    assert "No reference lab tests found" in output


# ---------------------------------------------------------------------------
# Live tests — only run with --run-live flag
# ---------------------------------------------------------------------------

@pytest.mark.skipif(True, reason="Live test — run with: pytest tests/test_m5.py --run-live")
def test_live_arup_group_a_strep(run_live):
    if not run_live:
        pytest.skip("pass --run-live to enable")
    from finder.sources.labs import find_reference_labs
    results = find_reference_labs("Group A Strep", labs=["arup"])
    assert len(results) >= 1
    for r in results:
        assert r.lab_name == "ARUP Laboratories"
        assert r.data_source == "Lab test directory (not an FDA determination)"


@pytest.mark.skipif(True, reason="Live test — run with: pytest tests/test_m5.py --run-live")
def test_live_mayo_group_a_strep(run_live):
    if not run_live:
        pytest.skip("pass --run-live to enable")
    from finder.sources.labs import find_reference_labs
    results = find_reference_labs("Group A Strep", labs=["mayo"])
    assert len(results) >= 1
    for r in results:
        assert r.lab_name == "Mayo Clinic Laboratories"
