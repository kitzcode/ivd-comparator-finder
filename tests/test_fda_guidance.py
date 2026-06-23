"""
Step 3 tests: the FDA guidance corpus on the shared grounded_rag engine.

Offline tests seed the guidance store directly (no network). One real-data test
runs only if the guidance snapshot has been ingested
(python -c "from corpora.fda_guidance import ingest_media_id; ingest_media_id('71075')").
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grounded_rag import Chunk, Corpus
from grounded_rag.qa import ask_corpus
from corpora.fda_guidance import FDAGuidanceCorpus
from corpora.fda_guidance.sections import chunk_guidance, _classify_heading

GUIDANCE_SNAPSHOT = (
    Path(__file__).parent.parent / "data" / "cache" / "guidance_chunks" / "FDA-GUID-71075.json"
)


@pytest.fixture
def isolated_guidance_store(tmp_path, monkeypatch):
    from corpora.fda_guidance import store as s
    monkeypatch.setattr(s, "CHUNK_DIR", tmp_path)
    monkeypatch.setattr(s, "MANIFEST_PATH", tmp_path / "_manifest.json")
    monkeypatch.setattr(s, "_COMMITTED_MANIFEST", tmp_path / "_manifest.json")
    monkeypatch.setattr(s, "_READ_DIRS", [tmp_path])
    return s


def _seed(s, doc_id, section, text, page, title="Test Guidance"):
    s.store_chunks(doc_id, [Chunk(
        doc_id=doc_id, source_url=f"https://www.fda.gov/media/{doc_id}/download",
        section=section, text=text, page=page, label=title,
        metadata={"corpus": "fda_guidance"},
    )])


# ---------------------------------------------------------------------------
# Corpus protocol + retrieval
# ---------------------------------------------------------------------------

def test_guidance_corpus_satisfies_protocol():
    c = FDAGuidanceCorpus()
    assert isinstance(c, Corpus)
    assert c.name == "fda_guidance"
    assert c.grounding.cited_id_pattern == r"FDA-GUID-\d+"


def test_keyword_answer_grounded(isolated_guidance_store):
    s = isolated_guidance_store
    _seed(s, "FDA-GUID-99001", "IV. Investigational Studies",
          "An IVD study is exempt from most IDE requirements when it meets the "
          "criteria in 21 CFR 812.2(c).", 18)
    result = ask_corpus(FDAGuidanceCorpus(), "When is a study exempt from IDE requirements?")
    assert "812.2" in result.answer
    assert result.citations[0].doc_id == "FDA-GUID-99001"
    assert result.citations[0].page == 18
    assert result.citations[0].label == "Test Guidance"


def test_llm_citation_uses_guidance_tag(isolated_guidance_store):
    s = isolated_guidance_store
    _seed(s, "FDA-GUID-99002", "III. General Regulatory Issues",
          "RUO products are not intended for clinical diagnostic use.", 7)
    llm = lambda sys, usr: "Per FDA-GUID-99002 (page 7), RUO products are not for diagnostic use."
    result = ask_corpus(FDAGuidanceCorpus(), "What are RUO products?", llm=llm)
    assert [c.doc_id for c in result.citations] == ["FDA-GUID-99002"]


def test_llm_refusal(isolated_guidance_store):
    s = isolated_guidance_store
    _seed(s, "FDA-GUID-99003", "I. Background", "Unrelated background text.", 1)
    llm = lambda sys, usr: (
        "The provided guidance documents do not contain sufficient information to answer this question."
    )
    result = ask_corpus(FDAGuidanceCorpus(), "What is the reimbursement rate?", llm=llm)
    assert result.not_found_reason is not None
    assert result.answer == ""


# ---------------------------------------------------------------------------
# Section splitter
# ---------------------------------------------------------------------------

def test_classify_heading_roman_and_appendix():
    assert _classify_heading("III. General Regulatory Issues") == "III. General Regulatory Issues"
    assert _classify_heading("Appendix 1: Regulatory Decision Tree").startswith("Appendix 1")
    assert _classify_heading("This is an ordinary sentence that runs on and on and on.") is None


def test_classify_heading_named():
    assert _classify_heading("Glossary") == "Glossary"
    assert _classify_heading("References") == "References"


# ---------------------------------------------------------------------------
# Real-data test (skipped unless the guidance snapshot is ingested)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not GUIDANCE_SNAPSHOT.exists(),
    reason="guidance snapshot not ingested; run ingest_media_id('71075')",
)
def test_real_guidance_same_engine_grounded():
    """The SAME engine answers over real guidance data with a real citation."""
    result = ask_corpus(
        FDAGuidanceCorpus(),
        "When is an IVD study exempt from most IDE requirements?",
        scope={"doc_ids": ["FDA-GUID-71075"]},
    )
    assert result.answer
    assert result.citations
    assert result.citations[0].doc_id == "FDA-GUID-71075"
    assert result.citations[0].page is not None
