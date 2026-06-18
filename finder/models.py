"""
Core pydantic models for the IVD Comparator Finder.
All output objects are read-only value types; no side effects on construction.
"""

from __future__ import annotations

from datetime import date
from typing import Optional
from pydantic import BaseModel, Field


class Device(BaseModel):
    """A cleared IVD device returned from openFDA 510(k)."""

    k_number: str
    device_name: str
    applicant_name: str
    decision_date: Optional[date] = None
    product_code: str
    regulation_number: Optional[str] = None
    device_class: Optional[str] = None
    # Best-effort link to the 510(k) Summary PDF; None if not yet resolved
    summary_url: Optional[str] = None
    # K-number of the predicate device cited in this clearance (if present)
    predicate_k_number: Optional[str] = None
    predicate_device_name: Optional[str] = None


class ProductCodeInfo(BaseModel):
    """Metadata for a three-letter openFDA product code."""

    product_code: str
    device_name: str
    regulation_number: Optional[str] = None
    device_class: Optional[str] = None
    medical_specialty: Optional[str] = None
    definition: Optional[str] = None


class AnalyteResolution(BaseModel):
    """The result of resolving an analyte term to product codes."""

    analyte_term: str
    synonyms_used: list[str]
    product_codes: list[ProductCodeInfo]
    # Heuristic warning; always surface to the user
    note: str = (
        "Product code mapping is heuristic (synonym text search). "
        "Review the synonym set and product code list before relying on completeness."
    )


class SummaryChunk(BaseModel):
    """A text chunk extracted from a 510(k) Summary PDF."""

    k_number: str
    product_code: str
    section: str  # e.g. "Intended Use", "Performance", "Limitations"
    text: str
    source_url: str
    page: Optional[int] = None


class Citation(BaseModel):
    """A grounded source citation."""

    k_number: str
    source_url: str
    page: Optional[int] = None
    section: Optional[str] = None


class Answer(BaseModel):
    """A grounded answer to a question about a device or product code scope."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    # If the source material doesn't support an answer, this is populated
    not_found_reason: Optional[str] = None
