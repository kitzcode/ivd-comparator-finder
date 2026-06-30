"""
Core tests for grounded_rag, exercised with a NON-FDA synthetic corpus.

These prove the reasoning layer is corpus-agnostic: no K-numbers, no PDFs, no
FDA types. Same engine, made-up "field guide" documents.
"""

from __future__ import annotations

from grounded_rag import (
    Answer,
    Chunk,
    GroundingContract,
    RetrievalConfig,
    answer,
    rank,
)

# A synthetic, non-FDA corpus: entries from a made-up field guide.
DOCS = [
    Chunk(
        doc_id="DOC-OAK",
        source_url="https://example.test/oak",
        section="Identification",
        text="The northern oak has lobed leaves and produces acorns in autumn.",
        page=12,
        label="Northern Oak",
    ),
    Chunk(
        doc_id="DOC-PINE",
        source_url="https://example.test/pine",
        section="Identification",
        text="The coastal pine has long needles in bundles of three and woody cones.",
        page=34,
        label="Coastal Pine",
    ),
    Chunk(
        doc_id="DOC-FERN",
        source_url="https://example.test/fern",
        section="Habitat",
        text="The shade fern thrives in damp, low-light forest understory.",
        page=7,
        label="Shade Fern",
    ),
]

CONFIG = RetrievalConfig(
    section_boost={"Identification": 2.0},
    domain_terms={"acorns": 3.0, "needles": 3.0},
)

CONTRACT = GroundingContract(
    system_prompt="Answer only from the field-guide context. If unknown, say NOT IN GUIDE.",
    not_found_sentinel="NOT IN GUIDE",
    id_leak_pattern=r"DOC-[A-Z]+",
)


def test_rank_orders_by_relevance():
    ranked = rank("acorns lobed leaves", DOCS, CONFIG, top_k=3)
    assert ranked[0].doc_id == "DOC-OAK"


def test_rank_section_filter():
    ranked = rank("forest", DOCS, CONFIG, top_k=3, sections=["Habitat"])
    assert all(c.section == "Habitat" for c in ranked)
    assert ranked[0].doc_id == "DOC-FERN"


def test_keyword_mode_returns_verbatim_snippet_with_citation():
    ranked = rank("needles cones", DOCS, CONFIG, top_k=3)
    result = answer("Describe the pine", ranked, contract=CONTRACT, config=CONFIG)
    assert isinstance(result, Answer)
    assert "needles" in result.answer
    assert result.citations[0].doc_id == "DOC-PINE"
    assert result.citations[0].page == 34
    assert result.not_found_reason is None


def test_llm_mode_index_based_citation():
    """Model cites by index; code attaches the real doc_id and a snippet."""
    ranked = rank("acorns", DOCS, CONFIG, top_k=3)  # DOC-OAK ranks first
    llm = lambda sys, usr: "The northern oak produces acorns [1].\nSUPPORTING: [1]"
    result = answer("oak?", ranked, contract=CONTRACT, config=CONFIG, llm=llm)
    assert [c.doc_id for c in result.citations] == ["DOC-OAK"]
    assert result.citations[0].snippet  # supporting text attached by code
    assert "[1]" in result.answer  # index marker preserved, no raw id in prose


def test_llm_mode_refusal_sets_not_found():
    ranked = rank("acorns", DOCS, CONFIG, top_k=3)
    llm = lambda sys, usr: "NOT IN GUIDE"
    result = answer("price of oak lumber?", ranked, contract=CONTRACT, config=CONFIG, llm=llm)
    assert result.not_found_reason is not None
    assert result.answer == ""
    assert result.citations == []


def test_only_selected_indices_are_cited():
    """A candidate the model did not select must not appear as a citation."""
    ranked = rank("acorns needles", DOCS, CONFIG, top_k=3)
    llm = lambda sys, usr: "Acorns come from one species [1].\nSUPPORTING: [1]"
    result = answer("compare", ranked, contract=CONTRACT, config=CONFIG, llm=llm)
    assert [c.doc_id for c in result.citations] == [ranked[0].doc_id]


def test_leak_guard_refuses_when_model_emits_identifier():
    """If the model writes a raw identifier, the answer is blanked and refused."""
    ranked = rank("acorns", DOCS, CONFIG, top_k=3)
    llm = lambda sys, usr: "See DOC-OAK [1].\nSUPPORTING: [1]"  # DOC-OAK is a leak
    result = answer("oak?", ranked, contract=CONTRACT, config=CONFIG, llm=llm)
    assert result.answer == ""
    assert result.citations == []
    assert result.not_found_reason is not None


def test_out_of_range_index_is_dropped():
    """The model cannot cite a candidate that was not retrieved."""
    ranked = rank("acorns", DOCS, CONFIG, top_k=3)  # 3 candidates exist
    llm = lambda sys, usr: "Answer [9].\nSUPPORTING: [9]"
    result = answer("oak?", ranked, contract=CONTRACT, config=CONFIG, llm=llm)
    assert result.citations == []
    assert result.not_found_reason is not None


def test_empty_scope_refuses():
    result = answer("anything", [], contract=CONTRACT, config=CONFIG)
    assert result.not_found_reason is not None
    assert result.answer == ""
