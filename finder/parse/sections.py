"""
Split a 510(k) Summary PDF's text into named sections.

Standard 510(k) Summary sections (21 CFR 807.92):
  - Device Description / Intended Use
  - Substantial Equivalence (SE) Discussion
  - Performance Testing / Summary of Testing
  - Conclusions / Limitations / Contraindications

Section boundaries are detected by heading patterns. The match is
intentionally broad because FDA summaries vary in heading style across
applicants and years. Every chunk that doesn't match a named section is
placed in a catch-all "Other" section so no text is silently dropped.
"""

from __future__ import annotations

import re
from typing import Optional

from ..models import SummaryChunk
from .pdf import PDFContent, PageContent

# ---------------------------------------------------------------------------
# Section heading patterns (case-insensitive; first match wins per line)
# ---------------------------------------------------------------------------

_SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Intended Use / Device Description", re.compile(
        r"(intended\s+use|device\s+description|indications?\s+for\s+use|"
        r"device\s+overview|device\s+name)", re.I
    )),
    ("Substantial Equivalence", re.compile(
        r"(substantial\s+equivalen|comparison\s+to\s+predicate|"
        r"predicate\s+device|se\s+comparison|technological\s+characteristics)", re.I
    )),
    ("Performance Testing", re.compile(
        r"(performance\s+testing|performance\s+data|clinical\s+performance|"
        r"analytical\s+performance|summary\s+of\s+(testing|studies|clinical)|"
        r"clinical\s+studies|sensitivity|specificity|ppa|npa|lod\b|"
        r"limit\s+of\s+detection|accuracy|precision|reproducib)", re.I
    )),
    ("Conclusions / Limitations", re.compile(
        r"(conclusion|limitation|contraindication|warning|caution|"
        r"interfering\s+substance|cross.react)", re.I
    )),
    ("Device Description", re.compile(
        r"(description\s+of\s+device|design\s+description|product\s+description|"
        r"components?)", re.I
    )),
]

_HEADING_LINE = re.compile(
    r"^\s{0,4}([A-Z][A-Za-z ,/\-\(\)]{4,80})\s*$"
)


def _classify_line(line: str) -> Optional[str]:
    """Return a section name if this line looks like a section heading."""
    if not _HEADING_LINE.match(line):
        return None
    for name, pat in _SECTION_PATTERNS:
        if pat.search(line):
            return name
    return None


# ---------------------------------------------------------------------------
# Chunking strategy
# ---------------------------------------------------------------------------

# Max characters per chunk (before splitting on paragraph boundaries)
_MAX_CHUNK_CHARS = 2000
_MIN_CHUNK_CHARS = 100


def _split_to_chunks(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split text into chunks of at most max_chars, breaking on blank lines."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    chunks: list[str] = []
    paragraphs = re.split(r"\n{2,}", text)
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > max_chars and current:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return [c for c in chunks if c.strip()]


def chunk_pdf(
    pdf: PDFContent,
    product_code: str,
    source_url: str,
) -> list[SummaryChunk]:
    """
    Convert a PDFContent into a list of SummaryChunks, one per section-chunk.

    Strategy:
    1. Walk pages in order, accumulating lines.
    2. When a heading is detected, flush the current section buffer and start
       a new section.
    3. At end of document, flush the final section.
    4. Each section's text is split into ≤2000-char chunks.
    5. Chunks below _MIN_CHUNK_CHARS are skipped (likely page headers/footers).
    """
    if pdf.is_image_only:
        return []

    chunks: list[SummaryChunk] = []
    current_section = "Other"
    current_lines: list[str] = []
    current_page = 1

    def flush(section: str, lines: list[str], page: int) -> None:
        text = "\n".join(lines).strip()
        if not text:
            return
        for i, chunk_text in enumerate(_split_to_chunks(text)):
            if len(chunk_text) < _MIN_CHUNK_CHARS:
                continue
            chunks.append(SummaryChunk(
                k_number=pdf.k_number,
                product_code=product_code,
                section=section,
                text=chunk_text,
                source_url=source_url,
                page=page,
            ))

    for page in pdf.pages:
        lines = page.text.splitlines()
        for line in lines:
            detected = _classify_line(line)
            if detected and detected != current_section:
                flush(current_section, current_lines, current_page)
                current_section = detected
                current_lines = []
                current_page = page.page_number
            else:
                current_lines.append(line)

        # Also emit table text as part of the current section
        for table in page.tables:
            ttext = table.to_text()
            if ttext.strip():
                current_lines.append("\n[TABLE]\n" + ttext + "\n[/TABLE]")

    flush(current_section, current_lines, current_page)
    return chunks
