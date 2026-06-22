"""
Generic value types for the grounded-RAG core. No FDA-specific fields.

A `Chunk` is the unit of retrieval: a passage of text with a stable document
id, the source it came from, and a free-form metadata bag for corpus-specific
fields (e.g. product_code, document_type) that the core never interprets.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrievable passage from some source document, corpus-agnostic.

    `doc_id` is the stable identifier of the source document within its corpus
    (a 510(k) K-number, a guidance media id, a filename — the core does not
    care which). `metadata` carries corpus-specific fields the core ignores.
    """

    doc_id: str
    source_url: str
    section: str
    text: str
    page: Optional[int] = None
    # Human-facing label for the document (e.g. "K173653" or a guidance title).
    label: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """A grounded source citation produced by the core, never by the model."""

    doc_id: str
    source_url: str
    page: Optional[int] = None
    section: Optional[str] = None
    label: Optional[str] = None


class Answer(BaseModel):
    """A grounded answer. If the sources don't support one, `answer` is empty
    and `not_found_reason` explains the refusal."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    not_found_reason: Optional[str] = None
