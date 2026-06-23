"""
M6 tests: MCP servers (two-layer split) — tool registration, annotations, and
the 10 eval questions against the cached snapshot.

  - mcp_servers.openfda_device — data layer (find_devices, get_clearance)
  - mcp_servers.grounded_rag   — reasoning layer (list_corpora, ask, compare_performance)

Eval questions use the cached snapshot (K173653, K141757, K201269) as the
known-answer substrate. They do not make live network calls.
"""

from __future__ import annotations

from pathlib import Path
import pytest

CHUNK_CACHE = Path(__file__).parent.parent / "data" / "cache" / "chunks"


def _indexed(*ks) -> bool:
    return all((CHUNK_CACHE / f"{k}.json").exists() for k in ks)


# ---------------------------------------------------------------------------
# Server imports, registration, and annotations
# ---------------------------------------------------------------------------

def test_servers_import_cleanly():
    import mcp_servers.openfda_device.server as dev
    import mcp_servers.grounded_rag.server as rag
    assert hasattr(dev, "mcp")
    assert hasattr(rag, "mcp")


def test_device_server_tools_registered():
    import mcp_servers.openfda_device.server as dev
    names = {t.name for t in dev.mcp._tool_manager.list_tools()}
    assert names == {"find_devices", "get_clearance"}, f"Registered: {names}"


def test_rag_server_tools_registered():
    import mcp_servers.grounded_rag.server as rag
    names = {t.name for t in rag.mcp._tool_manager.list_tools()}
    assert names == {"list_corpora", "ask", "compare_performance"}, f"Registered: {names}"


def test_all_tools_are_read_only():
    import mcp_servers.openfda_device.server as dev
    import mcp_servers.grounded_rag.server as rag
    for server in (dev.mcp, rag.mcp):
        for tool in server._tool_manager.list_tools():
            ann = tool.annotations
            assert ann is not None, f"{tool.name} has no annotations"
            assert ann.readOnlyHint is True, f"{tool.name}.readOnlyHint is not True"
            assert ann.destructiveHint is False, f"{tool.name}.destructiveHint is not False"


def test_legacy_alias_still_launches_device_server():
    """python -m ivd_mcp must still resolve to the data-layer server."""
    import ivd_mcp.ivd_server as legacy
    names = {t.name for t in legacy.mcp._tool_manager.list_tools()}
    assert names == {"find_devices", "get_clearance"}


def test_find_devices_has_description():
    import mcp_servers.openfda_device.server as dev
    tools = {t.name: t for t in dev.mcp._tool_manager.list_tools()}
    desc = tools["find_devices"].description or ""
    assert "analyte" in desc.lower() or "device" in desc.lower()


def test_compare_performance_description_mentions_predicate_vs_comparator():
    import mcp_servers.grounded_rag.server as rag
    tools = {t.name: t for t in rag.mcp._tool_manager.list_tools()}
    desc = tools["compare_performance"].description or ""
    assert "PREDICATE" in desc or "predicate" in desc
    assert "COMPARATOR" in desc or "comparator" in desc


def test_list_corpora_advertises_both_corpora():
    from mcp_servers.grounded_rag.server import list_corpora
    result = list_corpora()
    assert "fda_510k" in result["corpora"]
    assert "fda_guidance" in result["corpora"]


# ---------------------------------------------------------------------------
# Eval Q1–Q10: known-answer questions using cached snapshot
# ---------------------------------------------------------------------------

# Q1
def test_eval_q1_find_devices_returns_k173653():
    from mcp_servers.openfda_device.server import find_devices
    result = find_devices("Group A Strep")
    k_numbers = [d["k_number"] for d in result["devices"]]
    assert "K173653" in k_numbers, f"K173653 not in {k_numbers[:10]}"


# Q2
def test_eval_q2_find_devices_does_not_drop_bcid2():
    from mcp_servers.openfda_device.server import find_devices
    result = find_devices("Group A Strep")
    k_numbers = [d["k_number"] for d in result["devices"]]
    assert "K193519" in k_numbers, "K193519 (BioFire BCID2) was dropped — multiplex filtering bug"


# Q3
def test_eval_q3_dedicated_devices_under_866_2680():
    from mcp_servers.openfda_device.server import find_devices
    result = find_devices("Group A Strep")
    reg_by_k = {d["k_number"]: d["regulation_number"] for d in result["devices"]}
    for k in ("K173653", "K141757"):
        if k in reg_by_k:
            assert "866.2680" in (reg_by_k[k] or ""), (
                f"{k} regulation is {reg_by_k[k]!r}, expected 866.2680"
            )


# Q4
def test_eval_q4_get_clearance_k173653_fields():
    from mcp_servers.openfda_device.server import get_clearance
    result = get_clearance("K173653")
    assert "error" not in result
    assert result["product_code"] == "PGX", f"product_code={result['product_code']}"
    assert "Strep" in result["device_name"] or "strep" in result["device_name"].lower()


# Q5
def test_eval_q5_get_clearance_unknown_k_returns_error():
    from mcp_servers.openfda_device.server import get_clearance
    result = get_clearance("K000000")
    assert "error" in result


# Q6
@pytest.mark.skipif(not _indexed("K173653"), reason="K173653 not indexed")
def test_eval_q6_ask_lod_k173653():
    from mcp_servers.grounded_rag.server import ask
    result = ask("What is the limit of detection?", corpus="fda_510k", k_numbers=["K173653"])
    assert result["answer"] is not None or result["not_found_reason"] is not None
    if result["answer"]:
        text = result["answer"].lower()
        assert any(w in text for w in ["lod", "limit", "concentration", "cells", "cfu", "%"])
        assert result["citations"]
        assert result["citations"][0]["doc_id"] == "K173653"


# Q7
@pytest.mark.skipif(not _indexed("K173653"), reason="K173653 not indexed")
def test_eval_q7_ask_cites_page():
    from mcp_servers.grounded_rag.server import ask
    result = ask("reactivity strains tested", corpus="fda_510k", k_numbers=["K173653"])
    for cit in result["citations"]:
        assert cit["page"] is not None, f"Citation missing page: {cit}"


# Q8
@pytest.mark.skipif(not _indexed("K173653", "K141757"), reason="K173653 and K141757 not indexed")
def test_eval_q8_compare_performance_lod_both_devices():
    from mcp_servers.grounded_rag.server import compare_performance
    result = compare_performance(["K173653", "K141757"])
    rows = {row["k_number"]: row for row in result["rows"]}
    for k in ("K173653", "K141757"):
        assert k in rows
        assert rows[k]["lod"] is not None, f"LoD not extracted for {k}"
        assert rows[k]["lod"]["citation"]["k_number"] == k


# Q9
@pytest.mark.skipif(not _indexed("K173653"), reason="K173653 not indexed")
def test_eval_q9_compare_performance_predicate_note_present():
    from mcp_servers.grounded_rag.server import compare_performance
    result = compare_performance(["K173653"])
    assert "PREDICATE" in result["predicate_note"] or "predicate" in result["predicate_note"]
    assert "COMPARATOR" in result["predicate_note"] or "comparator" in result["predicate_note"]


# Q10 / grounding contract — no hallucinated K-numbers
@pytest.mark.skipif(not _indexed("K173653"), reason="K173653 not indexed")
def test_eval_q10_grounding_cited_k_numbers_match_scope():
    from mcp_servers.grounded_rag.server import ask
    result = ask("sensitivity and specificity", corpus="fda_510k", k_numbers=["K173653"])
    for cit in result["citations"]:
        assert cit["doc_id"] == "K173653", (
            f"Citation references {cit['doc_id']} but scope was K173653"
        )
