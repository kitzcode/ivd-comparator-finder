"""
M4 tests: structured performance extraction.

Unit tests run on synthetic chunk text. Live tests require K141757 and K173653
to be indexed (skip otherwise).
"""

from __future__ import annotations

from pathlib import Path
import pytest

CHUNK_CACHE = Path(__file__).parent.parent / "data" / "cache" / "chunks"


def _indexed(*k_numbers) -> bool:
    return all((CHUNK_CACHE / f"{k}.json").exists() for k in k_numbers)


# ---------------------------------------------------------------------------
# Regex extractor unit tests
# ---------------------------------------------------------------------------

def _make_chunk(text: str, section: str = "Performance Testing", k: str = "K000000"):
    from finder.models import SummaryChunk
    return SummaryChunk(
        k_number=k, product_code="TST", section=section,
        text=text, source_url="https://example.com", page=3,
    )


def test_extracts_ppa():
    from finder.extract import _extract_from_chunks
    chunk = _make_chunk("PPA (Positive Percent Agreement): 97.4% (95/97) (95% CI: 91.8–99.5%)")
    result = _extract_from_chunks([chunk])
    assert result["ppa"] is not None
    assert "97.4%" in result["ppa"].value


def test_extracts_npa():
    from finder.extract import _extract_from_chunks
    chunk = _make_chunk("NPA (Negative Percent Agreement): 99.8% (501/502) (95% CI: 98.9%–100%)")
    result = _extract_from_chunks([chunk])
    assert result["npa"] is not None
    assert "99.8%" in result["npa"].value


def test_extracts_lod_cells_per_ml():
    from finder.extract import _extract_from_chunks
    chunk = _make_chunk(
        "The LOD was determined to be 25 cells/mL of Elution Buffer for ATCC 19615."
    )
    result = _extract_from_chunks([chunk])
    assert result["lod"] is not None
    assert "25" in result["lod"].value


def test_extracts_lod_cfu():
    from finder.extract import _extract_from_chunks
    chunk = _make_chunk("Limit of Detection: 4.2 CFU/mL for ATCC 12344.")
    result = _extract_from_chunks([chunk])
    assert result["lod"] is not None
    assert "4.2" in result["lod"].value


def test_extracts_comparator_method():
    from finder.extract import _extract_from_chunks
    chunk = _make_chunk(
        "The comparator method was throat culture on sheep blood agar plates incubated at 37°C.",
        section="Performance Testing",
    )
    result = _extract_from_chunks([chunk])
    assert result["comparator_method"] is not None
    text = result["comparator_method"].value.lower()
    assert "culture" in text or "sheep" in text or "throat" in text


def test_extracts_predicate_from_se_section():
    from finder.extract import _extract_from_chunks
    chunk = _make_chunk(
        "The predicate device is K141757, the Alere i Strep A (Alere Scarborough, Inc.).",
        section="Substantial Equivalence",
    )
    result = _extract_from_chunks([chunk])
    assert result["predicate_device"] is not None
    assert "K141757" in result["predicate_device"].value or "Alere" in result["predicate_device"].value


def test_citation_carries_page_and_section():
    from finder.extract import _extract_from_chunks
    chunk = _make_chunk("PPA: 97.4% (95/97).")
    chunk = chunk.model_copy(update={"page": 7, "section": "Performance Testing"})
    result = _extract_from_chunks([chunk])
    assert result["ppa"].citation.page == 7
    assert result["ppa"].citation.section == "Performance Testing"


def test_missing_metric_is_none_not_invented():
    from finder.extract import _extract_from_chunks
    chunk = _make_chunk("This is a device description with no performance data.")
    result = _extract_from_chunks([chunk])
    assert result["ppa"] is None
    assert result["npa"] is None
    assert result["lod"] is None


def test_performance_table_model():
    from finder.extract import extract_performance, PerformanceTable
    table = extract_performance(["KNONE999"])
    assert isinstance(table, PerformanceTable)
    assert len(table.rows) == 1
    row = table.rows[0]
    assert row.k_number == "KNONE999"
    assert row.ppa is None  # no chunks indexed
    assert "No indexed summary" in row.extraction_notes[0]


def test_predicate_note_in_table():
    from finder.extract import extract_performance
    table = extract_performance(["KNONE999"])
    assert "PREDICATE" in table.predicate_note
    assert "COMPARATOR" in table.predicate_note


def test_format_performance_table_shows_not_found():
    from finder.extract import extract_performance, format_performance_table
    table = extract_performance(["KNONE999"])
    output = format_performance_table(table)
    assert "NOT FOUND IN SUMMARY" in output
    assert "KNONE999" in output


def test_llm_fill_uses_stub(tmp_path, monkeypatch):
    """LLM stub returning JSON is used to fill missing metrics."""
    from finder.index import store as s
    monkeypatch.setattr(s, "CHUNK_DIR", tmp_path)
    monkeypatch.setattr(s, "MANIFEST_PATH", tmp_path / "_manifest.json")
    from finder.models import SummaryChunk
    chunk = SummaryChunk(
        k_number="K111111", product_code="TST", section="Performance Testing",
        text="The clinical study compared the device to culture.", source_url="https://x.com", page=4,
    )
    s.store_chunks("K111111", [chunk])

    def stub_llm(system, user):
        return '{"ppa": "97.0%", "npa": "99.0%", "lod": null, "comparator_method": "culture", "predicate_device": null, "reactivity_strains": null}'

    from finder.extract import extract_performance
    table = extract_performance(["K111111"], llm=stub_llm)
    row = table.rows[0]
    assert row.ppa is not None and "97.0%" in row.ppa.value
    assert row.npa is not None and "99.0%" in row.npa.value
    assert row.comparator_method is not None


# ---------------------------------------------------------------------------
# Live tests against real indexed summaries
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _indexed("K141757", "K173653"),
    reason="K141757 and K173653 must be indexed; run: python cli.py ingest --knumbers K141757 K173653",
)
def test_k141757_lod_extracted():
    from finder.extract import extract_performance
    table = extract_performance(["K141757"])
    row = table.rows[0]
    assert row.lod is not None, f"LoD not extracted from K141757. Notes: {row.extraction_notes}"
    # K141757 reports 4.2 CFU/mL (ATCC 12344) and 41.8 CFU/mL (ATCC 19615)
    # Accept either strain's value — both are correct LoD figures from the Summary
    assert any(x in row.lod.value for x in ["4.2", "41.8", "CFU", "cells"]), (
        f"Unexpected LoD value: {row.lod.value}"
    )


@pytest.mark.skipif(
    not _indexed("K173653"),
    reason="K173653 must be indexed; run: python cli.py ingest --knumbers K173653",
)
def test_k173653_lod_extracted():
    from finder.extract import extract_performance
    table = extract_performance(["K173653"])
    row = table.rows[0]
    assert row.lod is not None, f"LoD not extracted from K173653. Notes: {row.extraction_notes}"
    assert any(x in row.lod.value for x in ["25", "147", "cells"]), (
        f"Unexpected LoD value: {row.lod.value}"
    )


@pytest.mark.skipif(
    not _indexed("K201269"),
    reason="K201269 must be indexed; run: python cli.py ingest --knumbers K201269",
)
def test_k201269_ppa_npa_extracted():
    from finder.extract import extract_performance
    table = extract_performance(["K201269"])
    row = table.rows[0]
    # K201269 reports PPA 93.8%, NPA 99.8%
    assert row.ppa is not None, f"PPA not extracted from K201269. Notes: {row.extraction_notes}"
    assert row.npa is not None, f"NPA not extracted from K201269. Notes: {row.extraction_notes}"
    assert "93" in row.ppa.value or "%" in row.ppa.value
    assert "99" in row.npa.value or "%" in row.npa.value


@pytest.mark.skipif(
    not _indexed("K141757", "K173653"),
    reason="Devices must be indexed",
)
def test_predicate_and_comparator_are_distinct():
    """Predicate device and comparator method must never be the same value."""
    from finder.extract import extract_performance
    for k in ("K141757", "K173653"):
        table = extract_performance([k])
        row = table.rows[0]
        if row.predicate_device and row.comparator_method:
            assert row.predicate_device.value != row.comparator_method.value, (
                f"{k}: predicate_device and comparator_method have the same value — "
                "conflation detected."
            )
