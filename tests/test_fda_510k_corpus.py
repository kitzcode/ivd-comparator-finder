"""
Step 2 tests: the FDA 510(k) build as a grounded_rag.Corpus.

Proves the finder now sits ON the Corpus protocol: candidates come back as
generic Chunks, and ask_corpus drives the same grounded answer end to end.
"""

from __future__ import annotations

import pytest

from grounded_rag import Chunk, Corpus
from grounded_rag.qa import ask_corpus


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    from finder.index import store as s
    monkeypatch.setattr(s, "CHUNK_DIR", tmp_path)
    monkeypatch.setattr(s, "MANIFEST_PATH", tmp_path / "_manifest.json")
    monkeypatch.setattr(s, "_COMMITTED_MANIFEST", tmp_path / "_manifest.json")
    monkeypatch.setattr(s, "_READ_DIRS", [tmp_path])
    return s


def _seed(s, k_number, product_code, section, text, page):
    from finder.models import SummaryChunk
    s.store_chunks(k_number, [SummaryChunk(
        k_number=k_number, product_code=product_code, section=section,
        text=text, source_url=f"https://fda.gov/{k_number}.pdf", page=page,
    )])


def test_corpus_satisfies_protocol():
    from corpora.fda_510k import FDA510kCorpus
    c = FDA510kCorpus()
    assert isinstance(c, Corpus)
    assert c.name == "fda_510k"
    assert c.grounding.cited_id_pattern == r"K\d{6}"


def test_candidates_return_generic_chunks(isolated_store):
    s = isolated_store
    _seed(s, "K200001", "TST", "Performance Testing", "PPA was 99.1%.", 3)
    from corpora.fda_510k import FDA510kCorpus
    chunks = FDA510kCorpus().candidates({"k_numbers": ["K200001"]})
    assert chunks and isinstance(chunks[0], Chunk)
    assert chunks[0].doc_id == "K200001"
    assert chunks[0].metadata["product_code"] == "TST"


def test_ask_corpus_keyword_mode_grounded(isolated_store):
    s = isolated_store
    _seed(s, "K200002", "AAA", "Performance Testing",
          "The LoD was 250 copies/mL.", 6)
    result = ask_corpus(
        FDA_corpus(),
        "What was the LoD?",
        scope={"k_numbers": ["K200002"]},
    )
    assert "250 copies/mL" in result.answer
    assert result.citations[0].doc_id == "K200002"
    assert result.citations[0].page == 6


def test_ask_corpus_llm_refusal(isolated_store):
    s = isolated_store
    _seed(s, "K200003", "BBB", "Other", "Unrelated boilerplate.", 1)
    llm = lambda sys, usr: (
        "The provided summaries do not contain sufficient information to answer this question."
    )
    result = ask_corpus(
        FDA_corpus(), "LoD?", scope={"k_numbers": ["K200003"]}, llm=llm,
    )
    assert result.not_found_reason is not None
    assert result.answer == ""


def FDA_corpus():
    from corpora.fda_510k import FDA510kCorpus
    return FDA510kCorpus()
