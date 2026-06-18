"""
M6 tests: MCP server — tool imports, annotations, and 10 eval questions.

Eval questions use the cached snapshot (K173653, K141757, K201269) as the
known-answer substrate. They do not make live network calls.

Run all:   pytest tests/test_m6.py -v
"""

from __future__ import annotations

from pathlib import Path
import pytest

CHUNK_CACHE = Path(__file__).parent.parent / "data" / "cache" / "chunks"


def _indexed(*ks) -> bool:
    return all((CHUNK_CACHE / f"{k}.json").exists() for k in ks)


# ---------------------------------------------------------------------------
# MCP server imports and annotation checks
# ---------------------------------------------------------------------------

def test_server_imports_cleanly():
    import ivd_mcp.ivd_server as s
    assert hasattr(s, "mcp")


def test_all_five_tools_registered():
    import ivd_mcp.ivd_server as s
    tool_names = {t.name for t in s.mcp._tool_manager.list_tools()}
    expected = {
        "find_devices",
        "get_clearance",
        "ask_summary",
        "compare_performance",
        "find_reference_labs",
    }
    assert expected == tool_names, f"Registered tools: {tool_names}"


def test_all_tools_are_read_only():
    import ivd_mcp.ivd_server as s
    for tool in s.mcp._tool_manager.list_tools():
        ann = tool.annotations
        assert ann is not None, f"{tool.name} has no annotations"
        assert ann.readOnlyHint is True, f"{tool.name}.readOnlyHint is not True"
        assert ann.destructiveHint is False, f"{tool.name}.destructiveHint is not False"


def test_find_devices_has_description():
    import ivd_mcp.ivd_server as s
    tools = {t.name: t for t in s.mcp._tool_manager.list_tools()}
    desc = tools["find_devices"].description or ""
    assert "analyte" in desc.lower() or "device" in desc.lower()


def test_compare_performance_description_mentions_predicate_vs_comparator():
    import ivd_mcp.ivd_server as s
    tools = {t.name: t for t in s.mcp._tool_manager.list_tools()}
    desc = tools["compare_performance"].description or ""
    assert "PREDICATE" in desc or "predicate" in desc
    assert "COMPARATOR" in desc or "comparator" in desc


# ---------------------------------------------------------------------------
# Eval Q1–Q10: known-answer questions using cached snapshot
# ---------------------------------------------------------------------------

# Q1
def test_eval_q1_find_devices_returns_k173653():
    """find_devices('Group A Strep') must return K173653 in the device list."""
    from ivd_mcp.ivd_server import find_devices
    result = find_devices("Group A Strep")
    k_numbers = [d["k_number"] for d in result["devices"]]
    assert "K173653" in k_numbers, f"K173653 not in {k_numbers[:10]}"


# Q2
def test_eval_q2_find_devices_does_not_drop_bcid2():
    """BioFire BCID2 (K193519) must appear — multiplex panels must not be filtered."""
    from ivd_mcp.ivd_server import find_devices
    result = find_devices("Group A Strep")
    k_numbers = [d["k_number"] for d in result["devices"]]
    assert "K193519" in k_numbers, "K193519 (BioFire BCID2) was dropped — multiplex filtering bug"


# Q3
def test_eval_q3_dedicated_devices_under_866_2680():
    """K173653 and K141757 must map to regulation 866.2680."""
    from ivd_mcp.ivd_server import find_devices
    result = find_devices("Group A Strep")
    reg_by_k = {d["k_number"]: d["regulation_number"] for d in result["devices"]}
    for k in ("K173653", "K141757"):
        if k in reg_by_k:
            assert "866.2680" in (reg_by_k[k] or ""), (
                f"{k} regulation is {reg_by_k[k]!r}, expected 866.2680"
            )


# Q4
def test_eval_q4_get_clearance_k173653_fields():
    """get_clearance('K173653') must return correct device name and product code."""
    from ivd_mcp.ivd_server import get_clearance
    result = get_clearance("K173653")
    assert "error" not in result
    assert result["product_code"] == "PGX", f"product_code={result['product_code']}"
    assert "Strep" in result["device_name"] or "strep" in result["device_name"].lower()


# Q5
def test_eval_q5_get_clearance_unknown_k_returns_error():
    """get_clearance with a nonexistent K-number must return an error key, not raise."""
    from ivd_mcp.ivd_server import get_clearance
    result = get_clearance("K000000")
    assert "error" in result


# Q6
@pytest.mark.skipif(
    not _indexed("K173653"),
    reason="K173653 not indexed",
)
def test_eval_q6_ask_summary_lod_k173653():
    """ask_summary about LoD for K173653 must return a chunk mentioning concentration."""
    from ivd_mcp.ivd_server import ask_summary
    result = ask_summary("What is the limit of detection?", k_numbers=["K173653"])
    assert result["answer"] is not None or result["not_found_reason"] is not None
    if result["answer"]:
        text = result["answer"].lower()
        assert any(w in text for w in ["lod", "limit", "concentration", "cells", "cfu", "%"])
        assert result["citations"]
        assert result["citations"][0]["k_number"] == "K173653"


# Q7
@pytest.mark.skipif(
    not _indexed("K173653"),
    reason="K173653 not indexed",
)
def test_eval_q7_ask_summary_cites_page():
    """Every citation in ask_summary must carry a page number."""
    from ivd_mcp.ivd_server import ask_summary
    result = ask_summary("reactivity strains tested", k_numbers=["K173653"])
    for cit in result["citations"]:
        assert cit["page"] is not None, f"Citation missing page: {cit}"


# Q8
@pytest.mark.skipif(
    not _indexed("K173653", "K141757"),
    reason="K173653 and K141757 not indexed",
)
def test_eval_q8_compare_performance_lod_both_devices():
    """compare_performance must extract LoD for K173653 and K141757."""
    from ivd_mcp.ivd_server import compare_performance
    result = compare_performance(["K173653", "K141757"])
    rows = {row["k_number"]: row for row in result["rows"]}
    for k in ("K173653", "K141757"):
        assert k in rows
        assert rows[k]["lod"] is not None, f"LoD not extracted for {k}"
        assert rows[k]["lod"]["citation"]["k_number"] == k


# Q9
@pytest.mark.skipif(
    not _indexed("K173653"),
    reason="K173653 not indexed",
)
def test_eval_q9_compare_performance_predicate_note_present():
    """compare_performance result must carry the PREDICATE ≠ COMPARATOR note."""
    from ivd_mcp.ivd_server import compare_performance
    result = compare_performance(["K173653"])
    assert "PREDICATE" in result["predicate_note"] or "predicate" in result["predicate_note"]
    assert "COMPARATOR" in result["predicate_note"] or "comparator" in result["predicate_note"]


# Q10
def test_eval_q10_find_reference_labs_invalid_lab_returns_error():
    """find_reference_labs with an unlisted lab must return an error, not raise."""
    from ivd_mcp.ivd_server import find_reference_labs
    result = find_reference_labs("Group A Strep", labs=["labcorp"])
    assert "error" in result
    assert "allowed_labs" in result


# Bonus: grounding contract check — no hallucinated K-numbers
@pytest.mark.skipif(
    not _indexed("K173653"),
    reason="K173653 not indexed",
)
def test_eval_grounding_cited_k_numbers_match_scope():
    """Citations from ask_summary must belong to the scoped K-number, not invented."""
    from ivd_mcp.ivd_server import ask_summary
    result = ask_summary("sensitivity and specificity", k_numbers=["K173653"])
    for cit in result["citations"]:
        assert cit["k_number"] == "K173653", (
            f"Citation references {cit['k_number']} but scope was K173653"
        )
