"""
The headline test: ONE engine, TWO corpora, the anti-hallucination contract
holding identically on both.

Each test is parametrized over the FDA 510(k) corpus and the FDA guidance corpus.
Both run through the exact same grounded_rag.qa.ask_corpus path. If the contract
ever diverges between corpora, one of these fails.

Contract proven on both:
  1. refuse-on-no-match  — the refusal sentinel sets not_found and blanks the answer.
  2. ground-with-citation — keyword mode returns a real citation (doc id + page).
  3. model-never-writes-a-citation — a document the model did not reference is
     never cited; only referenced documents appear.
"""

from __future__ import annotations

import pytest

from grounded_rag.qa import ask_corpus

# Both corpora use the same refusal phrase, so one stub serves both.
_REFUSAL = "The provided sources do not contain sufficient information to answer this question."


@pytest.fixture
def both_stores(tmp_path, monkeypatch):
    """Isolate both corpus stores into tmp dirs so tests never touch real cache."""
    from finder.index import store as s510
    from corpora.fda_guidance import store as sg

    d510 = tmp_path / "510k"
    dg = tmp_path / "guidance"
    d510.mkdir()
    dg.mkdir()

    for s, d in ((s510, d510), (sg, dg)):
        monkeypatch.setattr(s, "CHUNK_DIR", d)
        monkeypatch.setattr(s, "MANIFEST_PATH", d / "_manifest.json")
        monkeypatch.setattr(s, "_COMMITTED_MANIFEST", d / "_manifest.json")
        monkeypatch.setattr(s, "_READ_DIRS", [d])
    return s510, sg


def _seed_510k(s, doc_id, text, page):
    from finder.models import SummaryChunk
    s.store_chunks(doc_id, [SummaryChunk(
        k_number=doc_id, product_code="TST", section="Performance Testing",
        text=text, source_url=f"https://fda.gov/{doc_id}.pdf", page=page,
    )])


def _seed_guidance(s, doc_id, text, page):
    from grounded_rag.models import Chunk
    s.store_chunks(doc_id, [Chunk(
        doc_id=doc_id, source_url=f"https://www.fda.gov/media/{doc_id}/download",
        section="III. General Regulatory Issues", text=text, page=page,
        label="Test Guidance", metadata={"corpus": "fda_guidance"},
    )])


# Each case: how to build the corpus, seed two docs, and scope a query over them.
CASES = {
    "fda_510k": {
        "ids": ("K900001", "K900002"),
        "scope": {"k_numbers": ["K900001", "K900002"]},
    },
    "fda_guidance": {
        "ids": ("FDA-GUID-90001", "FDA-GUID-90002"),
        "scope": {"doc_ids": ["FDA-GUID-90001", "FDA-GUID-90002"]},
    },
}


def _make_corpus(name):
    if name == "fda_510k":
        from corpora.fda_510k import FDA510kCorpus
        return FDA510kCorpus()
    from corpora.fda_guidance import FDAGuidanceCorpus
    return FDAGuidanceCorpus()


def _seed_case(name, stores, id0, id1):
    s510, sg = stores
    if name == "fda_510k":
        _seed_510k(s510, id0, "The PPA was 99.0% against culture.", 5)
        _seed_510k(s510, id1, "Unrelated boilerplate about packaging.", 2)
    else:
        _seed_guidance(sg, id0, "The PPA was 99.0% against culture.", 5)
        _seed_guidance(sg, id1, "Unrelated boilerplate about packaging.", 2)


@pytest.mark.parametrize("corpus_name", list(CASES))
def test_refuses_on_no_match(corpus_name, both_stores):
    case = CASES[corpus_name]
    id0, id1 = case["ids"]
    _seed_case(corpus_name, both_stores, id0, id1)
    corpus = _make_corpus(corpus_name)

    result = ask_corpus(
        corpus, "What is the reimbursement rate?",
        scope=case["scope"], llm=lambda sysp, usr: _REFUSAL,
    )
    assert result.not_found_reason is not None
    assert result.answer == ""
    assert result.citations == []


@pytest.mark.parametrize("corpus_name", list(CASES))
def test_grounds_with_citation_keyword_mode(corpus_name, both_stores):
    case = CASES[corpus_name]
    id0, id1 = case["ids"]
    _seed_case(corpus_name, both_stores, id0, id1)
    corpus = _make_corpus(corpus_name)

    result = ask_corpus(corpus, "What was the PPA?", scope=case["scope"])
    assert "99.0%" in result.answer
    assert result.citations[0].doc_id == id0
    assert result.citations[0].page == 5


@pytest.mark.parametrize("corpus_name", list(CASES))
def test_model_selects_by_index_code_attaches_id(corpus_name, both_stores):
    """The model cites the first candidate by index; code attaches the real id.
    id0 (the PPA chunk) ranks first, so index 1 maps to it. id1 is never cited."""
    case = CASES[corpus_name]
    id0, id1 = case["ids"]
    _seed_case(corpus_name, both_stores, id0, id1)
    corpus = _make_corpus(corpus_name)

    llm = lambda sysp, usr: "The PPA was 99.0% [1].\nSUPPORTING: [1]"
    result = ask_corpus(corpus, "PPA?", scope=case["scope"], llm=llm)
    cited = [c.doc_id for c in result.citations]
    assert cited == [id0]
    assert id1 not in cited
    assert result.citations[0].snippet  # code-attached supporting text


@pytest.mark.parametrize("corpus_name", list(CASES))
def test_leak_guard_holds_on_both_corpora(corpus_name, both_stores):
    """If the model emits a real identifier, both corpora blank-and-refuse."""
    case = CASES[corpus_name]
    id0, id1 = case["ids"]
    _seed_case(corpus_name, both_stores, id0, id1)
    corpus = _make_corpus(corpus_name)

    llm = lambda sysp, usr: f"The PPA was 99.0% per {id0} [1].\nSUPPORTING: [1]"
    result = ask_corpus(corpus, "PPA?", scope=case["scope"], llm=llm)
    assert result.answer == ""
    assert result.citations == []
    assert result.not_found_reason is not None
