"""
Split a 510(k) Summary PDF's text into named sections and bounded chunks.

Standard 510(k) Summary sections (21 CFR 807.92):
  - Device Description / Intended Use
  - Substantial Equivalence (SE) Discussion
  - Performance Testing / Summary of Testing
  - Conclusions / Limitations / Contraindications

Section boundaries are detected by heading patterns. The match is
intentionally broad because FDA summaries vary in heading style across
applicants and years. Every line that doesn't match a named section is
placed in a catch-all "Other" section so no text is silently dropped.

Two classes of noise are removed before chunking:
  - The FDA decision/clearance cover letter (regulatory boilerplate).
  - Form FDA 3881 footer + the Paperwork Reduction Act burden statement.
Both pollute retrieval with text that is never an answer to a performance
question. The device's intended-use text itself is preserved.
"""

from __future__ import annotations

import re
from typing import Optional

from ..models import SummaryChunk
from .pdf import PDFContent

# ---------------------------------------------------------------------------
# Section heading patterns (case-insensitive; first match wins per line)
# ---------------------------------------------------------------------------

_SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Performance Testing", re.compile(
        r"(performance\s+(testing|characteristic|data|evaluation)|"
        r"analytical\s+(performance|sensitivity|specificity|study|studies)|"
        r"clinical\s+(performance|study|studies|evaluation)|"
        r"limit\s+of\s+detection|\blod\b|\bloq\b|"
        r"(positive|negative)\s+percent\s+agreement|\bppa\b|\bnpa\b|"
        r"reproducibility|repeatability|\bprecision\b|cross[\-\s]?reactivit|"
        r"interfering\s+substance|inclusivity|exclusivity|"
        r"method\s+comparison|summary\s+of\s+(testing|studies|performance))", re.I
    )),
    ("Substantial Equivalence", re.compile(
        r"(substantial\s+equivalen|comparison\s+(to|with|of)\s+(the\s+)?predicate|"
        r"predicate\s+(device|comparison)|technological\s+characteristic|"
        r"\bse\s+(comparison|discussion))", re.I
    )),
    ("Conclusions / Limitations", re.compile(
        r"(conclusion|\blimitation|contraindication|warnings?\s+and|"
        r"\bprecautions?\b)", re.I
    )),
    ("Intended Use / Device Description", re.compile(
        r"(intended\s+use|indications?\s+for\s+use|device\s+description|"
        r"device\s+overview|description\s+of\s+(the\s+)?device|"
        r"product\s+description|principle\s+of\s+(the\s+)?(test|procedure|operation))", re.I
    )),
]

# A line is a candidate heading if it is short and not a full sentence.
# Allows leading numbers/bullets ("5.", "5.1", "A.") and ALL-CAPS headings.
_HEADING_LINE = re.compile(
    r"^\s{0,6}"
    r"(?:[0-9]+(?:\.[0-9]+)*\.?\s+|[A-Z]\.\s+|[•\-\*]\s+)?"   # optional numbering/bullet
    r"([A-Za-z][A-Za-z0-9 ,/&\-\(\)]{3,70})\s*:?\s*$"
)


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    if not _HEADING_LINE.match(line):
        return False
    # Headings rarely end with a period (other than numbering) or contain
    # sentence-like trailing punctuation.
    if stripped.endswith(".") and not re.search(r"[0-9]\.$", stripped):
        return False
    return True


def _classify_line(line: str) -> Optional[str]:
    """Return a section name if this line looks like a section heading."""
    if not _looks_like_heading(line):
        return None
    for name, pat in _SECTION_PATTERNS:
        if pat.search(line):
            return name
    return None


# ---------------------------------------------------------------------------
# Boilerplate detection — drop FDA cover letter + PRA/Form 3881 noise
# ---------------------------------------------------------------------------

_BOILERPLATE_MARKERS = [
    "paperwork reduction act",
    "form fda 3881",
    "burden time for this collection",
    "prastaff@fda.hhs.gov",
    "an agency may not conduct or sponsor",
    "office of chief information officer",
    "psc publishing services",
    "omb control",
    "this section applies only to requirements",
    "department of health and human services",
    "send comments regarding this burden",
    "currently valid omb",
]

# FDA decision-letter phrases (the regulatory cover letter before the Summary).
_DECISION_LETTER_MARKERS = [
    "we have reviewed your section 510(k)",
    "we have reviewed your premarket notification",
    "substantially equivalent to legally marketed",
    "general controls provisions of the act",
    "you may, therefore, market the device",
    "this letter will allow you to begin marketing",
    "center for devices and radiological health",
    "premarket notification submission",
    "the food and drug administration",
]


def _is_boilerplate_line(line: str) -> bool:
    low = line.lower()
    return any(m in low for m in _BOILERPLATE_MARKERS)


def _chunk_is_boilerplate(text: str) -> bool:
    """True if the chunk is dominated by FDA cover-letter / PRA boilerplate."""
    low = text.lower()
    marker_hits = sum(1 for m in (_BOILERPLATE_MARKERS + _DECISION_LETTER_MARKERS) if m in low)
    if marker_hits >= 3:
        return True
    # Short chunk that is purely a form footer / address block
    if marker_hits >= 1 and len(text) < 400:
        return True
    return False


# ---------------------------------------------------------------------------
# Chunking strategy
# ---------------------------------------------------------------------------

_MAX_CHUNK_CHARS = 1400
_MIN_CHUNK_CHARS = 80


def _pack_lines(
    entries: list[tuple[str, int]],
    max_chars: int = _MAX_CHUNK_CHARS,
) -> list[tuple[str, int]]:
    """
    Pack (line, page) entries into chunks no larger than max_chars.

    Unlike a blank-line splitter, this is line-based: it works even when the
    PDF text has no double-newlines (the common case from pdfplumber). A blank
    line is a soft preferred break; an over-long single line is hard-split.

    Returns a list of (chunk_text, page) where page is the page of the chunk's
    first line — so citations point at where the text actually is.
    """
    chunks: list[tuple[str, int]] = []
    cur: list[str] = []
    cur_len = 0
    cur_page: Optional[int] = None

    def flush() -> None:
        nonlocal cur, cur_len, cur_page
        if cur:
            text = "\n".join(cur).strip()
            if text:
                chunks.append((text, cur_page if cur_page is not None else 1))
        cur, cur_len, cur_page = [], 0, None

    for line, page in entries:
        # Hard-split pathologically long single lines (rare; wide tables).
        while len(line) > max_chars:
            flush()
            chunks.append((line[:max_chars], page))
            line = line[max_chars:]

        is_blank = not line.strip()

        # Start a new chunk when adding this line would overflow.
        if cur_len + len(line) + 1 > max_chars and cur:
            flush()

        if cur_page is None and not is_blank:
            cur_page = page

        cur.append(line)
        cur_len += len(line) + 1

        # A blank line at a comfortable size is a clean break point.
        if is_blank and cur_len > max_chars * 0.6:
            flush()

    flush()
    return chunks


def chunk_pdf(
    pdf: PDFContent,
    product_code: str,
    source_url: str,
) -> list[SummaryChunk]:
    """
    Convert a PDFContent into a list of SummaryChunks.

    Strategy:
    1. Walk pages in order, tracking each line's page number.
    2. Drop FDA cover-letter / PRA boilerplate lines.
    3. When a heading is detected, start a new section.
    4. Pack each section's lines into <=_MAX_CHUNK_CHARS chunks (line-based,
       so it works even without blank-line separators).
    5. Drop chunks below _MIN_CHUNK_CHARS or dominated by boilerplate.
    6. Each chunk's page = the page of its first line.
    """
    if pdf.is_image_only:
        return []

    chunks: list[SummaryChunk] = []
    current_section = "Other"
    current_entries: list[tuple[str, int]] = []  # (line, page)

    def flush_section(section: str, entries: list[tuple[str, int]]) -> None:
        if not entries:
            return
        for chunk_text, page in _pack_lines(entries):
            if len(chunk_text) < _MIN_CHUNK_CHARS:
                continue
            if _chunk_is_boilerplate(chunk_text):
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
        for line in page.text.splitlines():
            if _is_boilerplate_line(line):
                continue
            detected = _classify_line(line)
            if detected and detected != current_section:
                flush_section(current_section, current_entries)
                current_section = detected
                current_entries = [(line, page.page_number)]
            else:
                current_entries.append((line, page.page_number))

        # Append table text to the current section, tagged to this page.
        for table in page.tables:
            ttext = table.to_text()
            if ttext.strip():
                current_entries.append(("\n[TABLE]\n" + ttext + "\n[/TABLE]", page.page_number))

    flush_section(current_section, current_entries)
    return chunks
