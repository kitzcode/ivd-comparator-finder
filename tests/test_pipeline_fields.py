"""
Regression for the openFDA 510(k) field-mapping defect found in Phase 1 recon
and verified against cached responses (see RECON.md):
  - applicant lives in `applicant`, not `applicant_name`
  - decision date lives in `decision_date`, not `decision_date_as_string`;
    `date_received` is the submission date and must not be reported as the decision
  - `traditional_501k_flag` is a Y/N flag, not a predicate K-number
"""

from __future__ import annotations

from finder.pipeline import _normalize_device


def _rec(**over) -> dict:
    base = {
        "k_number": "K173653",
        "device_name": "Example Strep A Test",
        "applicant": "Acme Diagnostics, Inc.",
        "decision_date": "2018-03-15",
        "date_received": "2017-11-01",
        "decision_code": "SESE",
        "device_class": "2",
        "traditional_501k_flag": "N",
    }
    base.update(over)
    return base


def test_applicant_mapped_from_real_field():
    d = _normalize_device(_rec(), "PGX", "866.2680")
    assert d.applicant_name == "Acme Diagnostics, Inc."


def test_decision_date_is_decision_not_received():
    d = _normalize_device(_rec(), "PGX", "866.2680")
    assert str(d.decision_date) == "2018-03-15"  # not the 2017-11-01 submission date


def test_decision_date_falls_back_to_received_only_when_absent():
    d = _normalize_device(_rec(decision_date=None), "PGX", "866.2680")
    assert str(d.decision_date) == "2017-11-01"


def test_flag_not_mislabeled_as_predicate():
    d = _normalize_device(_rec(), "PGX", "866.2680")
    assert d.predicate_k_number is None


def test_denovo_detected_from_den_number():
    d = _normalize_device(_rec(k_number="DEN140005", decision_code="DENG"), "OEY", "866.3375")
    assert d.submission_type == "De Novo"
