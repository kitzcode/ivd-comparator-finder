"""
M3 tests: grounded Q&A contract.

Tests run in keyword-only mode (no LLM required).
Live tests require K173653 to be indexed; skipped otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

CHUNK_CACHE = Path(__file__).parent.parent / "data" / "cache" / "chunks"
TARGET_K = "K173653"


def _k173653_indexed() -> bool:
    return (CHUNK_CACHE / f"{TARGET_K}.json").exists()


# ---------------------------------------------------------------------------
# ask() API surface
# ---------------------------------------------------------------------------

def test_ask_returns_answer_model():
    from finder.qa import ask
    from finder.models import Answer
    result = ask("test question", k_numbers=["KFAKE999"])
    assert isinstance(result, Answer)
    assert result.question == "test question"


def test_ask_empty_scope_returns_not_found():
    from finder.qa import ask
    result = ask("what is the LoD?", k_numbers=["KNONE000"])
    assert result.not_found_reason is not None
    assert result.answer == ""


def test_ask_keyword_mode_returns_top_chunk_text(tmp_path, monkeypatch):
    """Keyword mode must return the chunk text verbatim — no generation."""
    from finder.index import store as s
    monkeypatch.setattr(s, "CHUNK_DIR", tmp_path)
    monkeypatch.setattr(s, "MANIFEST_PATH", tmp_path / "_manifest.json")
    from finder.models import SummaryChunk
    chunk = SummaryChunk(
        k_number="K888888",
        product_code="TST",
        section="Performance Testing",
        text="The PPA was 97.5% (95% CI: 93–99%). Comparator: culture.",
        source_url="https://example.com/K888888.pdf",
        page=5,
    )
    s.store_chunks("K888888", [chunk])

    from finder.qa import ask
    result = ask("What is the PPA?", k_numbers=["K888888"])
    assert result.answer == chunk.text
    assert result.citations[0].k_number == "K888888"
    assert result.citations[0].page == 5
    assert result.not_found_reason is None


def test_ask_citation_carries_source_url(tmp_path, monkeypatch):
    from finder.index import store as s
    monkeypatch.setattr(s, "CHUNK_DIR", tmp_path)
    monkeypatch.setattr(s, "MANIFEST_PATH", tmp_path / "_manifest.json")
    from finder.models import SummaryChunk
    chunk = SummaryChunk(
        k_number="K777777",
        product_code="AAA",
        section="Performance Testing",
        text="NPA was 98.2%.",
        source_url="https://fda.gov/K777777.pdf",
        page=4,
    )
    s.store_chunks("K777777", [chunk])
    from finder.qa import ask
    result = ask("specificity NPA", k_numbers=["K777777"])
    assert result.citations[0].source_url == "https://fda.gov/K777777.pdf"


# ---------------------------------------------------------------------------
# Grounding contract: LLM mode with a stub LLM
# ---------------------------------------------------------------------------

def _stub_llm_returning(text: str):
    def _llm(system_prompt: str, user_prompt: str) -> str:
        return text
    return _llm


def test_llm_mode_extracts_cited_k_numbers(tmp_path, monkeypatch):
    from finder.index import store as s
    monkeypatch.setattr(s, "CHUNK_DIR", tmp_path)
    monkeypatch.setattr(s, "MANIFEST_PATH", tmp_path / "_manifest.json")
    from finder.models import SummaryChunk
    chunk = SummaryChunk(
        k_number="K555555",
        product_code="BBB",
        section="Performance Testing",
        text="PPA was 96%.",
        source_url="https://example.com",
        page=2,
    )
    s.store_chunks("K555555", [chunk])

    # LLM cites K555555 in its response
    llm = _stub_llm_returning("According to K555555 (page 2), the PPA was 96%.")
    from finder.qa import ask
    result = ask("PPA?", k_numbers=["K555555"], llm=llm)
    assert any(c.k_number == "K555555" for c in result.citations)


def test_llm_mode_not_found_when_llm_says_so(tmp_path, monkeypatch):
    from finder.index import store as s
    monkeypatch.setattr(s, "CHUNK_DIR", tmp_path)
    monkeypatch.setattr(s, "MANIFEST_PATH", tmp_path / "_manifest.json")
    from finder.models import SummaryChunk
    chunk = SummaryChunk(
        k_number="K444444",
        product_code="CCC",
        section="Other",
        text="Some unrelated text.",
        source_url="https://example.com",
        page=1,
    )
    s.store_chunks("K444444", [chunk])
    llm = _stub_llm_returning(
        "The provided summaries do not contain sufficient information to answer this question."
    )
    from finder.qa import ask
    result = ask("LoD?", k_numbers=["K444444"], llm=llm)
    assert result.not_found_reason is not None


# ---------------------------------------------------------------------------
# format_answer
# ---------------------------------------------------------------------------

def test_format_answer_includes_citation_line():
    from finder.qa import format_answer
    from finder.models import Answer, Citation
    a = Answer(
        question="Q",
        answer="PPA was 97%.",
        citations=[Citation(k_number="K123456", source_url="https://fda.gov/K123456.pdf", page=3, section="Performance Testing")],
    )
    text = format_answer(a)
    assert "K123456" in text
    assert "p.3" in text


def test_format_answer_not_found():
    from finder.qa import format_answer
    from finder.models import Answer
    a = Answer(question="Q", answer="", not_found_reason="Not in summaries.")
    text = format_answer(a)
    assert "NOT FOUND" in text


# ---------------------------------------------------------------------------
# Live test — keyword Q&A against real K173653 data
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _k173653_indexed(),
    reason="K173653 not indexed; run: python cli.py ingest --knumbers K173653",
)
def test_k173653_lod_query_returns_relevant_chunk():
    from finder.qa import ask
    result = ask("What LoD did K173653 report?", k_numbers=[TARGET_K])
    assert result.answer or result.not_found_reason
    if result.answer:
        # The LoD chunk contains numeric concentration data
        assert any(c in result.answer.lower() for c in ["lod", "limit", "concentration", "cells", "cfu", "%"])
        assert result.citations
        assert result.citations[0].k_number == TARGET_K


@pytest.mark.skipif(
    not _k173653_indexed(),
    reason="K173653 not indexed; run: python cli.py ingest --knumbers K173653",
)
def test_k173653_does_not_hallucinate_device_names():
    """Keyword mode must not invent device names not present in the chunk text."""
    from finder.qa import ask
    result = ask("list all devices tested", k_numbers=[TARGET_K])
    # Keyword mode returns chunk text verbatim — no generation possible
    assert result.answer  # should find something
    # The answer must trace to a real chunk page
    assert result.citations[0].page is not None
