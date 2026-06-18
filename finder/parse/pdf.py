"""
PDF → raw text + table extraction using pdfplumber.

Returns a list of PageContent objects, each with the page's plain text
and any tables found. Tables are returned as list[list[str]] (rows of cells).

Caller is responsible for detecting image-only PDFs before calling here;
use summaries.is_image_only_pdf() to check first.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class TableRow(BaseModel):
    cells: list[Optional[str]]


class ExtractedTable(BaseModel):
    rows: list[TableRow]
    bbox: Optional[tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)

    def to_text(self) -> str:
        """Render table as a simple pipe-delimited string for chunking."""
        lines = []
        for row in self.rows:
            cells = [str(c) if c is not None else "" for c in row.cells]
            lines.append(" | ".join(cells))
        return "\n".join(lines)


class PageContent(BaseModel):
    page_number: int  # 1-indexed
    text: str
    tables: list[ExtractedTable] = []

    @property
    def full_text(self) -> str:
        """Combined text + tables for indexing."""
        parts = [self.text]
        for t in self.tables:
            parts.append(t.to_text())
        return "\n".join(p for p in parts if p.strip())


class PDFContent(BaseModel):
    k_number: str
    source_path: str
    pages: list[PageContent]
    is_image_only: bool = False
    page_count: int = 0

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.full_text for p in self.pages)


def extract_pdf(pdf_path: Path, k_number: str) -> PDFContent:
    """
    Extract text and tables from all pages of a 510(k) Summary PDF.

    Returns a PDFContent with per-page text and tables.
    If the PDF is image-only (no extractable text), sets is_image_only=True
    and returns empty pages — the caller should flag this for OCR.
    """
    import pdfplumber

    pages: list[PageContent] = []
    image_only = False

    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        total_chars = 0

        for i, page in enumerate(pdf.pages):
            raw_text = page.extract_text() or ""
            total_chars += len(raw_text.strip())

            # Extract tables with pdfplumber's default strategy
            raw_tables = page.extract_tables() or []
            tables: list[ExtractedTable] = []
            for raw_table in raw_tables:
                if not raw_table:
                    continue
                rows = [TableRow(cells=[str(c) if c is not None else "" for c in row]) for row in raw_table]
                tables.append(ExtractedTable(rows=rows))

            pages.append(PageContent(
                page_number=i + 1,
                text=raw_text,
                tables=tables,
            ))

        # Flag image-only if we got almost no text across the whole document
        if page_count > 0 and total_chars / page_count < 20:
            image_only = True

    return PDFContent(
        k_number=k_number,
        source_path=str(pdf_path),
        pages=pages,
        is_image_only=image_only,
        page_count=page_count,
    )
