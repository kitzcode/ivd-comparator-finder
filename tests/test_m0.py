"""
M0 tests: models, analyte synonym resolution, and openFDA client (import-clean).
These run without network or cache.
"""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def _modules():
    return [
        ROOT / "finder" / "models.py",
        ROOT / "finder" / "analyte.py",
        ROOT / "finder" / "sources" / "openfda.py",
        ROOT / "finder" / "sources" / "summaries.py",
        ROOT / "finder" / "pipeline.py",
        ROOT / "cli.py",
    ]


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_py_compile_clean(path):
    """Every module must compile without syntax errors."""
    py_compile.compile(str(path), doraise=True)


def test_models_import():
    from finder.models import Device, ProductCodeInfo, AnalyteResolution, SummaryChunk, Citation, Answer
    d = Device(k_number="K123456", device_name="Test", applicant_name="Acme", product_code="AAA")
    assert d.k_number == "K123456"


def test_analyte_resolution_model():
    from finder.models import AnalyteResolution, ProductCodeInfo
    res = AnalyteResolution(
        analyte_term="Group A Strep",
        synonyms_used=["Group A Strep", "GAS"],
        product_codes=[ProductCodeInfo(product_code="MOM", device_name="Strep A test")],
    )
    assert len(res.product_codes) == 1
    assert "heuristic" in res.note.lower()


def test_builtin_synonyms_for_gas():
    from finder.analyte import get_synonyms
    syns = get_synonyms("Group A Strep")
    assert "Streptococcus pyogenes" in syns
    assert "GAS" in syns
    assert len(syns) >= 4


def test_extra_synonyms_merged():
    from finder.analyte import get_synonyms
    syns = get_synonyms("Group A Strep", extra_synonyms=["Custom Term"])
    assert "Custom Term" in syns


def test_unknown_analyte_falls_back_to_term():
    from finder.analyte import get_synonyms
    syns = get_synonyms("Esoteric Pathogen XYZ")
    assert "Esoteric Pathogen XYZ" in syns
