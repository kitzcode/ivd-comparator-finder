"""
Adversarial / contract suite, run through grounded_rag.qa.ask_corpus on BOTH
corpora. These try to break the anti-hallucination contract: leak an identifier,
cite a source that was not retrieved, or answer something unsupported.

Written against CONTRACTS.md section 3 (index-based selection + three guards).
"""

from __future__ import annotations

import pytest

from grounded_rag.qa import ask_corpus


@pytest.fixture
def both_stores(tmp_path, monkeypatch):
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


CASES = {
    "fda_510k": {"id": "K900001", "scope": {"k_numbers": ["K900001"]}, "leak": "K404040"},
    "fda_guidance": {"id": "FDA-GUID-90001", "scope": {"doc_ids": ["FDA-GUID-90001"]},
                     "leak": "FDA-GUID-40404"},
}


def _make_corpus(name):
    if name == "fda_510k":
        from corpora.fda_510k import FDA510kCorpus
        return FDA510kCorpus()
    from corpora.fda_guidance import FDAGuidanceCorpus
    return FDAGuidanceCorpus()


def _seed_one(name, stores, doc_id, text="The PPA was 99.0% against culture.", page=5):
    s510, sg = stores
    (_seed_510k if name == "fda_510k" else _seed_guidance)(
        s510 if name == "fda_510k" else sg, doc_id, text, page)


# ---------------------------------------------------------------------------
# Guard 1: citation leakage (identifier and URL)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("corpus_name", list(CASES))
def test_leak_identifier_blanks_and_refuses(corpus_name, both_stores):
    case = CASES[corpus_name]
    _seed_one(corpus_name, both_stores, case["id"])
    leak = case["leak"]
    llm = lambda s, u: f"The PPA was 99.0% per {leak} [1].\nSUPPORTING: [1]"
    result = ask_corpus(_make_corpus(corpus_name), "PPA?", scope=case["scope"], llm=llm)
    assert result.answer == ""
    assert result.citations == []
    assert result.not_found_reason is not None


@pytest.mark.parametrize("corpus_name", list(CASES))
def test_leak_lowercase_identifier_blanks_and_refuses(corpus_name, both_stores):
    case = CASES[corpus_name]
    _seed_one(corpus_name, both_stores, case["id"])
    leak = case["leak"].lower()  # e.g. k404040 / fda-guid-40404
    llm = lambda s, u: f"The PPA was 99.0% per {leak} [1].\nSUPPORTING: [1]"
    result = ask_corpus(_make_corpus(corpus_name), "PPA?", scope=case["scope"], llm=llm)
    assert result.answer == ""
    assert result.citations == []
    assert result.not_found_reason is not None


@pytest.mark.parametrize("corpus_name", list(CASES))
def test_leak_hidden_in_supporting_line_refuses(corpus_name, both_stores):
    """An identifier hidden in the SUPPORTING line (stripped from view) is still a
    leak: the guard scans the full raw model output, not just the visible prose."""
    case = CASES[corpus_name]
    _seed_one(corpus_name, both_stores, case["id"])
    leak = case["leak"]
    llm = lambda s, u: f"The PPA was 99.0% [1].\nSUPPORTING: [1, {leak}]"
    result = ask_corpus(_make_corpus(corpus_name), "PPA?", scope=case["scope"], llm=llm)
    assert result.answer == ""
    assert result.citations == []
    assert result.not_found_reason is not None


@pytest.mark.parametrize("corpus_name", list(CASES))
def test_unicode_digit_index_is_not_accepted(corpus_name, both_stores):
    """A Unicode digit lookalike must not be parsed as a valid index."""
    case = CASES[corpus_name]
    _seed_one(corpus_name, both_stores, case["id"])
    # U+FF11 FULLWIDTH DIGIT ONE: str.isdigit() is True, but it is not ASCII.
    llm = lambda s, u: "The PPA was high [１].\nSUPPORTING: [１]"
    result = ask_corpus(_make_corpus(corpus_name), "PPA?", scope=case["scope"], llm=llm)
    assert result.citations == []
    assert result.not_found_reason is not None


@pytest.mark.parametrize("corpus_name", list(CASES))
def test_leak_url_blanks_and_refuses(corpus_name, both_stores):
    case = CASES[corpus_name]
    _seed_one(corpus_name, both_stores, case["id"])
    llm = lambda s, u: "The PPA was 99.0%, see https://example.com [1].\nSUPPORTING: [1]"
    result = ask_corpus(_make_corpus(corpus_name), "PPA?", scope=case["scope"], llm=llm)
    assert result.answer == ""
    assert result.citations == []
    assert result.not_found_reason is not None


# ---------------------------------------------------------------------------
# Guard 2: source-existence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("corpus_name", list(CASES))
def test_out_of_range_index_refuses(corpus_name, both_stores):
    case = CASES[corpus_name]
    _seed_one(corpus_name, both_stores, case["id"])  # only 1 candidate exists
    llm = lambda s, u: "The PPA was high [5].\nSUPPORTING: [5]"
    result = ask_corpus(_make_corpus(corpus_name), "PPA?", scope=case["scope"], llm=llm)
    assert result.citations == []
    assert result.not_found_reason is not None


@pytest.mark.parametrize("corpus_name", list(CASES))
def test_every_citation_is_in_retrieved_set(corpus_name, both_stores):
    case = CASES[corpus_name]
    _seed_one(corpus_name, both_stores, case["id"])
    llm = lambda s, u: "The PPA was 99.0% [1].\nSUPPORTING: [1]"
    result = ask_corpus(_make_corpus(corpus_name), "PPA?", scope=case["scope"], llm=llm)
    assert result.citations
    for c in result.citations:
        assert c.doc_id == case["id"]  # nothing cited outside the seeded scope


# ---------------------------------------------------------------------------
# Snippet presence (never a figure without its source text)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("corpus_name", list(CASES))
def test_keyword_mode_citation_has_snippet(corpus_name, both_stores):
    case = CASES[corpus_name]
    _seed_one(corpus_name, both_stores, case["id"])
    result = ask_corpus(_make_corpus(corpus_name), "What was the PPA?", scope=case["scope"])
    assert result.citations
    assert result.citations[0].snippet


# ---------------------------------------------------------------------------
# One engine, two corpora: the generalization proof in a single test
# ---------------------------------------------------------------------------

def test_one_engine_two_corpora(both_stores):
    _seed_one("fda_510k", both_stores, "K900001")
    _seed_one("fda_guidance", both_stores, "FDA-GUID-90001")
    stub = lambda s, u: "The PPA was 99.0% [1].\nSUPPORTING: [1]"

    r510 = ask_corpus(_make_corpus("fda_510k"), "PPA?",
                      scope=CASES["fda_510k"]["scope"], llm=stub)
    rgui = ask_corpus(_make_corpus("fda_guidance"), "PPA?",
                      scope=CASES["fda_guidance"]["scope"], llm=stub)

    for result, expected_id in ((r510, "K900001"), (rgui, "FDA-GUID-90001")):
        assert result.answer  # grounded answer, not a refusal
        assert result.citations[0].doc_id == expected_id
        assert result.citations[0].snippet
