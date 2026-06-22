"""
Split an FDA guidance PDF into named sections and bounded generic Chunks.

Guidance documents are structured differently from 510(k) Summaries (verified on
real PDFs): a Table of Contents, then top-level Roman-numeral sections
(I. Background, II. Introduction, ...), plus Appendices. We use the heading text
itself as the section label, so citations land on a meaningful section.

Noise dropped before chunking:
  - the per-page "Contains Nonbinding Recommendations" running header,
  - Table-of-Contents dot-leader lines (e.g. "III. General .... 7"),
  - bare page-number lines.

Output is grounded_rag.Chunk (doc_id = "FDA-GUID-{media_id}", label = title), so
the same engine that serves 510(k) chunks serves these.
"""

from __future__ import annotations

import re
from typing import Optional

from grounded_rag.models import Chunk
# Line-packing is generic substrate, shared with the 510(k) splitter.
from finder.parse.sections import _pack_lines
from finder.parse.pdf import PDFContent

_MIN_CHUNK_CHARS = 80

# Top-level Roman-numeral heading: "III. General Regulatory Issues"
_ROMAN_HEADING = re.compile(
    r"^\s*((?:IX|IV|V?I{0,3}|XI{0,3}|XI?V|XVI{0,3})\.)\s+([A-Z][A-Za-z0-9 ,/&:\-\(\)]{2,70})\s*$"
)
# Appendix heading: "Appendix 1: ..." / "Appendix A ..."
_APPENDIX_HEADING = re.compile(r"^\s*(Appendix\s+[0-9A-Z]+[:.]?\s*[A-Za-z0-9 ,/&\-\(\)]{0,70})\s*$")
# Named front/back-matter sections.
_NAMED_HEADING = re.compile(
    r"^\s*(Preface|Table of Contents|References|Glossary|Background|Introduction|Scope|Definitions)\s*$",
    re.I,
)

# A Table-of-Contents entry: text followed by dot leaders and a page number.
_TOC_LEADER = re.compile(r"\.{4,}\s*\d+\s*$")
_BARE_PAGE_NUM = re.compile(r"^\s*\d{1,4}\s*$")
_NONBINDING = "contains nonbinding recommendations"


def _classify_heading(line: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None
    m = _ROMAN_HEADING.match(line)
    if m:
        return f"{m.group(1)} {m.group(2).strip()}"
    m = _APPENDIX_HEADING.match(line)
    if m:
        return m.group(1).strip()
    m = _NAMED_HEADING.match(line)
    if m:
        return m.group(1).strip().title()
    return None


def _is_noise(line: str) -> bool:
    low = line.lower().strip()
    if not low:
        return False  # blank lines are soft break hints for the packer
    if _NONBINDING in low:
        return True
    if _TOC_LEADER.search(line):
        return True
    if _BARE_PAGE_NUM.match(line):
        return True
    return False


def chunk_guidance(
    pdf: PDFContent,
    *,
    doc_id: str,
    source_url: str,
    title: str,
) -> list[Chunk]:
    """Convert a parsed guidance PDF into grounded_rag.Chunks.

    Walks pages in order, drops running-header / TOC / page-number noise, starts
    a new section at each detected heading, and packs each section's lines into
    bounded chunks tagged with the page of their first line.
    """
    if pdf.is_image_only:
        return []

    chunks: list[Chunk] = []
    current_section = "Preamble"
    current_entries: list[tuple[str, int]] = []

    def flush(section: str, entries: list[tuple[str, int]]) -> None:
        if not entries:
            return
        for text, page in _pack_lines(entries):
            if len(text) < _MIN_CHUNK_CHARS:
                continue
            chunks.append(Chunk(
                doc_id=doc_id,
                source_url=source_url,
                section=section,
                text=text,
                page=page,
                label=title,
                metadata={"corpus": "fda_guidance"},
            ))

    for page in pdf.pages:
        for line in page.text.splitlines():
            if _is_noise(line):
                continue
            heading = _classify_heading(line)
            if heading and heading != current_section:
                flush(current_section, current_entries)
                current_section = heading
                current_entries = [(line, page.page_number)]
            else:
                current_entries.append((line, page.page_number))
        for table in page.tables:
            ttext = table.to_text()
            if ttext.strip():
                current_entries.append(("\n[TABLE]\n" + ttext + "\n[/TABLE]", page.page_number))

    flush(current_section, current_entries)
    return chunks
