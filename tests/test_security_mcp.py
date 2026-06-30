"""
Security boundary tests for the MCP tools: hostile or malformed identifiers must
be rejected with an actionable error before any lookup, fetch, or store access.
These run offline (validation precedes any network/filesystem use).
"""

from __future__ import annotations

from finder.security import is_valid_device_id, is_valid_guidance_id


def test_guidance_id_validator():
    assert is_valid_guidance_id("FDA-GUID-71075")
    assert not is_valid_guidance_id("../../etc/passwd")
    assert not is_valid_guidance_id("FDA-GUID-")
    assert not is_valid_guidance_id("")


def test_device_id_validator_rejects_traversal():
    assert is_valid_device_id("K173653")
    assert not is_valid_device_id("../../secret")
    assert not is_valid_device_id("K173653.json")


# ---------------------------------------------------------------------------
# openFDA device server
# ---------------------------------------------------------------------------

def test_get_clearance_rejects_invalid_id():
    from mcp_servers.openfda_device.server import get_clearance
    result = get_clearance("../../../../etc/hosts")
    assert "error" in result
    assert "Invalid device id" in result["error"]


def test_find_devices_rejects_overlong_analyte():
    from mcp_servers.openfda_device.server import find_devices
    result = find_devices("x" * 5000)
    assert "error" in result


# ---------------------------------------------------------------------------
# grounded_rag server
# ---------------------------------------------------------------------------

def test_ask_rejects_invalid_k_numbers():
    from mcp_servers.grounded_rag.server import ask
    result = ask("PPA?", corpus="fda_510k", k_numbers=["../../evil"])
    assert "error" in result
    assert "Invalid k_numbers" in result["error"]


def test_ask_rejects_invalid_doc_ids():
    from mcp_servers.grounded_rag.server import ask
    result = ask("scope?", corpus="fda_guidance", doc_ids=["../../../etc/passwd"])
    assert "error" in result
    assert "Invalid doc_ids" in result["error"]


def test_ask_rejects_too_many_ids():
    from mcp_servers.grounded_rag.server import ask
    result = ask("PPA?", corpus="fda_510k", k_numbers=[f"K{n:06d}" for n in range(100)])
    assert "error" in result
    assert "Too many" in result["error"]


def test_compare_performance_rejects_invalid_id():
    from mcp_servers.grounded_rag.server import compare_performance
    result = compare_performance(["../../evil"])
    assert "error" in result
    assert "Invalid device id" in result["error"]
