"""
M2 tests: PDF fetch, extraction, sectioning, chunking, and store/retrieve.

Tests that need a real PDF run against the cached K173653 PDF.
If the PDF is not yet cached, they are skipped (run `python cli.py ingest
--knumbers K173653` first to populate).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PDF_CACHE = Path(__file__).parent.parent / "data" / "cache" / "pdf"
CHUNK_CACHE = Path(__file__).parent.parent / "data" / "cache" / "chunks"
TARGET_K = "K173653"  # Alere i Strep A 2 — known to have a public Summary PDF


# ---------------------------------------------------------------------------
# summaries.py
# ---------------------------------------------------------------------------

def test_resolve_summary_url_caches_result(tmp_path, monkeypatch):
    """resolve_summary_url() writes a .url sidecar; second call reads from it."""
    from finder.sources import summaries
    # Patch both cache dirs so committed sidecars don't shadow the fake URL
    monkeypatch.setattr(summaries, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(summaries, "_COMMITTED_URL_DIR", tmp_path)
    url_file = tmp_path / f"{TARGET_K}.url"
    url_file.write_text("https://fake.example.com/K173653.pdf")
    result = summaries.resolve_summary_url(TARGET_K)
    assert result == "https://fake.example.com/K173653.pdf"


def test_resolve_summary_url_none_when_cached_as_missing(tmp_path, monkeypatch):
    from finder.sources import summaries
    monkeypatch.setattr(summaries, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(summaries, "_COMMITTED_URL_DIR", tmp_path)
    url_file = tmp_path / f"{TARGET_K}.url"
    url_file.write_text("NONE")
    result = summaries.resolve_summary_url(TARGET_K)
    assert result is None


# ---------------------------------------------------------------------------
# pdf.py — import-level and image-only detection
# ---------------------------------------------------------------------------

def test_pdf_module_imports():
    from finder.parse.pdf import extract_pdf, PDFContent, PageContent, ExtractedTable


def test_image_only_detection_empty_pdf(tmp_path):
    """A PDF with no text should be flagged as image-only."""
    # Create a minimal blank PDF using pdfplumber's underlying pdfminer
    # Instead, create a tiny valid PDF manually
    blank_pdf = tmp_path / "blank.pdf"
    blank_pdf.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
    )
    from finder.parse.pdf import extract_pdf
    result = extract_pdf(blank_pdf, "KTEST")
    assert result.is_image_only is True
    assert result.page_count == 1


# ---------------------------------------------------------------------------
# sections.py
# ---------------------------------------------------------------------------

def test_section_heading_classification():
    from finder.parse.sections import _classify_line
    assert _classify_line("Performance Testing Summary") == "Performance Testing"
    assert _classify_line("Intended Use") == "Intended Use / Device Description"
    assert _classify_line("Substantial Equivalence Discussion") == "Substantial Equivalence"
    assert _classify_line("Conclusions and Limitations") == "Conclusions / Limitations"
    # A body line should not match
    assert _classify_line("The sensitivity of the assay was 97.5%") is None


def test_chunk_pdf_empty_returns_empty():
    from finder.parse.pdf import PDFContent
    from finder.parse.sections import chunk_pdf
    pdf = PDFContent(k_number="K000000", source_path="/dev/null", pages=[], is_image_only=False, page_count=0)
    chunks = chunk_pdf(pdf, product_code="AAA", source_url="https://example.com")
    assert chunks == []


def test_chunk_pdf_image_only_returns_empty():
    from finder.parse.pdf import PDFContent
    from finder.parse.sections import chunk_pdf
    pdf = PDFContent(k_number="K000000", source_path="/dev/null", pages=[], is_image_only=True, page_count=1)
    chunks = chunk_pdf(pdf, product_code="AAA", source_url="https://example.com")
    assert chunks == []


def test_chunk_pdf_produces_chunks_for_real_text():
    from finder.parse.pdf import PDFContent, PageContent
    from finder.parse.sections import chunk_pdf
    page = PageContent(
        page_number=1,
        text=(
            "Intended Use\n"
            "This device is intended for the detection of Streptococcus pyogenes "
            "(Group A Strep) from throat swabs in symptomatic patients.\n\n"
            "Performance Testing\n"
            "The sensitivity (PPA) was 97.4% (95% CI: 93.8–99.1%).\n"
            "The specificity (NPA) was 98.2% (95% CI: 95.2–99.5%).\n"
        ),
        tables=[],
    )
    pdf = PDFContent(k_number="K173653", source_path="/test", pages=[page], page_count=1)
    chunks = chunk_pdf(pdf, "PGX", "https://example.com/K173653.pdf")
    assert len(chunks) >= 1
    sections = {c.section for c in chunks}
    assert "Performance Testing" in sections or "Intended Use / Device Description" in sections
    for c in chunks:
        assert c.k_number == "K173653"
        assert c.product_code == "PGX"
        assert c.source_url == "https://example.com/K173653.pdf"


# ---------------------------------------------------------------------------
# store.py
# ---------------------------------------------------------------------------

def test_store_and_load_roundtrip(tmp_path, monkeypatch):
    from finder.index import store as s
    monkeypatch.setattr(s, "CHUNK_DIR", tmp_path)
    monkeypatch.setattr(s, "MANIFEST_PATH", tmp_path / "_manifest.json")
    from finder.models import SummaryChunk
    chunks = [
        SummaryChunk(
            k_number="K999999",
            product_code="XYZ",
            section="Performance Testing",
            text="PPA was 97%",
            source_url="https://example.com",
            page=3,
        )
    ]
    s.store_chunks("K999999", chunks)
    loaded = s.load_chunks("K999999")
    assert len(loaded) == 1
    assert loaded[0].text == "PPA was 97%"
    assert loaded[0].page == 3
    assert s.is_indexed("K999999")
    assert s.get_index_status("K999999") == "ok"


# ---------------------------------------------------------------------------
# retrieve.py
# ---------------------------------------------------------------------------

def test_retrieve_scores_performance_section_higher(tmp_path, monkeypatch):
    from finder.index import store as s
    monkeypatch.setattr(s, "CHUNK_DIR", tmp_path)
    monkeypatch.setattr(s, "MANIFEST_PATH", tmp_path / "_manifest.json")
    from finder.models import SummaryChunk

    perf_chunk = SummaryChunk(
        k_number="K111111", product_code="AAA",
        section="Performance Testing",
        text="The sensitivity PPA was 97.4% and specificity NPA was 98.2%",
        source_url="https://example.com", page=3,
    )
    other_chunk = SummaryChunk(
        k_number="K111111", product_code="AAA",
        section="Other",
        text="The sensitivity PPA was 97.4% and specificity NPA was 98.2%",
        source_url="https://example.com", page=1,
    )
    s.store_chunks("K111111", [perf_chunk, other_chunk])

    from finder.index.retrieve import retrieve
    results = retrieve("sensitivity PPA NPA", k_numbers=["K111111"], top_k=2)
    assert len(results) >= 1
    # Performance Testing chunk should rank first due to section boost
    assert results[0].section == "Performance Testing"


# ---------------------------------------------------------------------------
# Live tests — require cached PDF (skip if absent)
# ---------------------------------------------------------------------------

def _k173653_pdf() -> Path:
    return PDF_CACHE / f"{TARGET_K}.pdf"


def _k173653_chunks() -> Path:
    return CHUNK_CACHE / f"{TARGET_K}.json"


@pytest.mark.skipif(
    not (_k173653_pdf()).exists(),
    reason="K173653 PDF not cached; run: python cli.py ingest --knumbers K173653",
)
def test_k173653_pdf_extraction_yields_text():
    from finder.parse.pdf import extract_pdf
    pdf = extract_pdf(_k173653_pdf(), TARGET_K)
    assert pdf.page_count > 0
    assert not pdf.is_image_only, "K173653 should not be image-only"
    assert len(pdf.full_text) > 500


@pytest.mark.skipif(
    not (_k173653_chunks()).exists(),
    reason="K173653 not indexed; run: python cli.py ingest --knumbers K173653",
)
def test_k173653_chunks_cover_performance_section():
    from finder.index.store import load_chunks
    chunks = load_chunks(TARGET_K)
    assert len(chunks) > 0
    sections = {c.section for c in chunks}
    assert "Performance Testing" in sections, (
        f"No Performance Testing section found in K173653. Sections present: {sections}"
    )


@pytest.mark.skipif(
    not (_k173653_chunks()).exists(),
    reason="K173653 not indexed; run: python cli.py ingest --knumbers K173653",
)
def test_k173653_retrieve_sensitivity():
    from finder.index.retrieve import retrieve
    results = retrieve("sensitivity specificity PPA NPA", k_numbers=[TARGET_K], top_k=5)
    assert len(results) > 0
    combined = " ".join(c.text for c in results).lower()
    # The K173653 summary should mention sensitivity or PPA somewhere
    assert "sensitiv" in combined or "ppa" in combined or "%" in combined
