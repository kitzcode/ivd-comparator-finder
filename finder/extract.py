"""
M4: Structured performance extraction.

Extracts PPA/NPA, LoD, reactivity strains, and comparator/reference method
from indexed SummaryChunks. Every extracted value carries a Citation so
the output table is fully auditable.

Design constraints:
  - Regex-first: extract from chunk text without an LLM when the pattern is
    unambiguous (PPA/NPA percentages, LoD concentrations).
  - LLM-assisted: when an LLM is supplied, it is scoped to a single chunk and
    asked to extract one thing at a time with the grounding contract enforced.
  - If a value is not in the chunks, the cell is None — never invented.
  - Predicate device and comparator/reference method are always kept separate.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from pydantic import BaseModel

from .models import Citation, SummaryChunk
from .index.retrieve import retrieve
from .index.store import load_chunks


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class PerformanceValue(BaseModel):
    """A single extracted performance metric with its provenance."""
    value: str              # e.g. "97.4% (95/97)" or "25 cells/mL"
    raw_text: str           # the chunk fragment the value was drawn from
    citation: Citation


class PerformanceRow(BaseModel):
    """All extracted performance data for one K-number."""
    k_number: str
    device_name: str
    product_code: str

    # Clinical performance (PPA = sensitivity, NPA = specificity)
    ppa: Optional[PerformanceValue] = None          # Positive Percent Agreement
    npa: Optional[PerformanceValue] = None          # Negative Percent Agreement

    # Analytical performance
    lod: Optional[PerformanceValue] = None          # Limit of Detection
    reactivity_strains: Optional[PerformanceValue] = None

    # Comparator / reference method used in the study
    comparator_method: Optional[PerformanceValue] = None

    # Predicate device cited for substantial equivalence (distinct from comparator)
    predicate_device: Optional[PerformanceValue] = None

    extraction_notes: list[str] = []


class PerformanceTable(BaseModel):
    """Comparison table across multiple devices."""
    rows: list[PerformanceRow]
    scope_note: str = (
        "All values extracted from 510(k) Summary PDFs. "
        "Each cell cites its source K-number and page. "
        "Empty cells mean the data was not found in the indexed summary, "
        "not that it was not studied."
    )
    predicate_note: str = (
        "PREDICATE ≠ COMPARATOR: the predicate device is cited for substantial "
        "equivalence; the comparator/reference method is what performance was "
        "measured against in the clinical study. These are distinct roles."
    )


# ---------------------------------------------------------------------------
# Regex extractors
# ---------------------------------------------------------------------------

# PPA / sensitivity — matches "97.4% (95/97)" or "97.4% (95% CI: ...)"
_PPA_PATTERNS = [
    re.compile(
        r"(?:PPA|[Ss]ensitivity|[Pp]ositive [Pp]ercent [Aa]greement)[^\d]{0,30}"
        r"(\d{1,3}(?:\.\d+)?%(?:[^\n]{0,60})?)",
        re.I,
    ),
    re.compile(
        r"(\d{1,3}(?:\.\d+)?%[^\n]{0,60})\s*\(?\s*(?:sensitivity|PPA)",
        re.I,
    ),
]

_NPA_PATTERNS = [
    re.compile(
        r"(?:NPA|[Ss]pecificity|[Nn]egative [Pp]ercent [Aa]greement)[^\d]{0,30}"
        r"(\d{1,3}(?:\.\d+)?%(?:[^\n]{0,60})?)",
        re.I,
    ),
    re.compile(
        r"(\d{1,3}(?:\.\d+)?%[^\n]{0,60})\s*\(?\s*(?:specificity|NPA)",
        re.I,
    ),
]

# LoD — matches "25 cells/mL" or "4.2 CFU/mL" or "147 cells/mL"
# Also handles table rows: "ATCC 12344 | 147 |  |  | 100%" in a LoD table context
_LOD_PATTERNS = [
    # Standard prose: "LOD was ... 25 cells/mL" — up to 200 chars gap
    re.compile(
        r"(?:LOD|[Ll]imit of [Dd]etection)[^\d]{0,200}"
        r"(\d+(?:\.\d+)?\s*(?:cells?|CFU|copies?|TCID50|PFU)?[/\s]*(?:mL|µL|uL|reaction|swab))",
        re.I | re.S,
    ),
    # Table row in a LoD table: "ATCC NNNNN | <number> |" where surrounding text
    # mentions LOD / Elution Buffer / concentration
    re.compile(
        r"(?:Elution Buffer|LOD|[Cc]oncentration)[^\n]{0,120}\n"
        r"(?:[^\n]*\n){0,3}"             # up to 3 rows of header
        r"[^\n]*ATCC[^\n]*\|\s*(\d+(?:\.\d+)?)\s*\|",
        re.I | re.S,
    ),
    # Trailing pattern: number then unit before LOD mention
    re.compile(
        r"(\d+(?:\.\d+)?\s*(?:cells?|CFU|copies?)[/\s]*mL)[^\n]{0,80}(?:LOD|limit|detect)",
        re.I,
    ),
]

# Comparator / reference method — culture, PCR, sequencing, etc.
_COMPARATOR_PATTERNS = [
    re.compile(
        r"(?:comparator|reference method|compared (?:to|against)|predicate method|"
        r"composite reference|comparator test|reference standard)[^\n]{0,8}[:\-–]?\s*"
        r"([A-Za-z][^\n]{5,120})",
        re.I,
    ),
    re.compile(
        r"(?:culture|PCR|sequencing|genotyping|NAAT)[^\n]{0,30}"
        r"(?:was used as|served as|as the)\s+(?:comparator|reference)",
        re.I,
    ),
]

# Reactivity — list of ATCC strains or named strains
_REACTIVITY_PATTERN = re.compile(
    r"(?:reactivity|strains? tested|inclusive|detected at or near)[^\n]{0,200}"
    r"((?:ATCC\s*\w+[,\s]+){2,}[ATCC\s\w]+)",
    re.I | re.S,
)

# Predicate device — mentioned in SE section
_PREDICATE_PATTERNS = [
    re.compile(
        r"(?:predicate device|predicates?|substantial equivalence to)[^\n]{0,8}[:\-–]?\s*"
        r"([A-Z][^\n]{5,100})",
        re.I,
    ),
    re.compile(r"(K\d{6})[^\n]{0,60}(?:predicate|predicated on|equivalent to)", re.I),
]


def _try_patterns(patterns: list[re.Pattern], text: str) -> Optional[str]:
    for pat in patterns:
        m = pat.search(text)
        if m:
            return m.group(1).strip()[:300]
    return None


def _make_perf_value(value_str: str, chunk: SummaryChunk) -> PerformanceValue:
    return PerformanceValue(
        value=value_str[:200],
        raw_text=chunk.text[:500],
        citation=Citation(
            k_number=chunk.k_number,
            source_url=chunk.source_url,
            page=chunk.page,
            section=chunk.section,
        ),
    )


# ---------------------------------------------------------------------------
# Per-device extraction
# ---------------------------------------------------------------------------

def _extract_from_chunks(
    chunks: list[SummaryChunk],
    llm: Optional[Callable[[str, str], str]] = None,
) -> dict[str, Optional[PerformanceValue]]:
    """
    Extract PPA, NPA, LoD, reactivity, comparator, and predicate from chunks.
    Returns a dict keyed by metric name.
    """
    results: dict[str, Optional[PerformanceValue]] = {
        "ppa": None, "npa": None, "lod": None,
        "reactivity_strains": None, "comparator_method": None,
        "predicate_device": None,
    }

    # Group by section for targeted extraction
    perf_chunks = [c for c in chunks if "performance" in c.section.lower() or "conclusion" in c.section.lower()]
    se_chunks = [c for c in chunks if "substantial" in c.section.lower()]
    all_chunks = chunks

    # Prose fields (free-text method / device names) where regex produces noisy,
    # mid-sentence matches. When an LLM is available, let it own these instead of
    # trusting the regex; the regex remains the fallback when no LLM is supplied.
    have_llm = llm is not None

    # --- Regex extraction pass ---
    for chunk in perf_chunks + all_chunks:
        text = chunk.text

        if results["ppa"] is None:
            val = _try_patterns(_PPA_PATTERNS, text)
            if val:
                results["ppa"] = _make_perf_value(val, chunk)

        if results["npa"] is None:
            val = _try_patterns(_NPA_PATTERNS, text)
            if val:
                results["npa"] = _make_perf_value(val, chunk)

        if results["lod"] is None:
            val = _try_patterns(_LOD_PATTERNS, text)
            if val:
                results["lod"] = _make_perf_value(val, chunk)

        if not have_llm and results["comparator_method"] is None:
            val = _try_patterns(_COMPARATOR_PATTERNS, text)
            if val:
                results["comparator_method"] = _make_perf_value(val, chunk)

        if not have_llm and results["reactivity_strains"] is None:
            m = _REACTIVITY_PATTERN.search(text)
            if m:
                results["reactivity_strains"] = _make_perf_value(m.group(1).strip()[:300], chunk)

    if not have_llm:
        for chunk in se_chunks + all_chunks:
            if results["predicate_device"] is None:
                val = _try_patterns(_PREDICATE_PATTERNS, chunk.text)
                if val:
                    results["predicate_device"] = _make_perf_value(val, chunk)

    # --- LLM extraction pass: fill prose fields + any numeric field still missing ---
    if have_llm:
        missing = [k for k, v in results.items() if v is None]
        if missing:
            # Include performance, SE and intended-use chunks so the model can
            # find predicate/comparator that live outside the perf sections.
            iu_chunks = [c for c in chunks if "intended" in c.section.lower()]
            context = (perf_chunks[:4] + se_chunks[:2] + iu_chunks[:1]) or all_chunks[:4]
            results = _llm_fill_missing(results, missing, context, llm)

    return results


_LLM_EXTRACT_SYSTEM = """\
You are extracting structured performance data from a 510(k) Summary PDF.

RULES:
1. Extract ONLY from the provided text chunk. Do not use prior knowledge.
2. For each field, return the value as it appears in the text, verbatim or minimally paraphrased.
3. If a field is not present in the chunk, return null for it.
4. PREDICATE ≠ COMPARATOR: the predicate device is cited for substantial equivalence; \
the comparator/reference method is what performance was measured against in the clinical study.
5. Do not invent numbers, device names, or K-numbers.

Respond with a JSON object with these keys (all optional, null if absent):
{
  "ppa": "97.4% (95% CI: ...)",
  "npa": "99.2% (95% CI: ...)",
  "lod": "25 cells/mL",
  "comparator_method": "throat culture on sheep blood agar",
  "predicate_device": "K141757 (Alere i Strep A)",
  "reactivity_strains": "ATCC 12344, ATCC 19615, ..."
}
"""


def _llm_fill_missing(
    results: dict,
    missing: list[str],
    chunks: list[SummaryChunk],
    llm: Callable[[str, str], str],
) -> dict:
    import json as _json

    combined_text = "\n\n".join(c.text[:900] for c in chunks[:6])
    prompt = (
        f"Extract the following fields from this 510(k) Summary text "
        f"(return null for any not present): {missing}\n\n"
        f"TEXT:\n{combined_text}"
    )
    try:
        raw = llm(_LLM_EXTRACT_SYSTEM, prompt)
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("`")
        parsed = _json.loads(raw)
    except Exception:
        return results

    # Use a representative chunk for citation
    ref_chunk = chunks[0]
    for key in missing:
        val = parsed.get(key)
        if val and isinstance(val, str) and val.strip():
            results[key] = _make_perf_value(val.strip(), ref_chunk)

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_performance(
    k_numbers: list[str],
    device_names: Optional[dict[str, str]] = None,
    product_codes: Optional[dict[str, str]] = None,
    llm: Optional[Callable[[str, str], str]] = None,
) -> PerformanceTable:
    """
    Extract structured performance data for each K-number and return a
    PerformanceTable with one row per device.

    device_names and product_codes are {k_number: value} lookup dicts;
    they are used to populate the row labels and are not required.
    """
    rows: list[PerformanceRow] = []

    for k in k_numbers:
        chunks = load_chunks(k)
        metrics = _extract_from_chunks(chunks, llm=llm)

        notes: list[str] = []
        if not chunks:
            notes.append("No indexed summary — run ingest first")
        missing_metrics = [m for m, v in metrics.items() if v is None]
        if missing_metrics:
            notes.append(f"Not found in summary: {', '.join(missing_metrics)}")

        rows.append(PerformanceRow(
            k_number=k,
            device_name=(device_names or {}).get(k, ""),
            product_code=(product_codes or {}).get(k, ""),
            ppa=metrics["ppa"],
            npa=metrics["npa"],
            lod=metrics["lod"],
            reactivity_strains=metrics["reactivity_strains"],
            comparator_method=metrics["comparator_method"],
            predicate_device=metrics["predicate_device"],
            extraction_notes=notes,
        ))

    return PerformanceTable(rows=rows)


def format_performance_table(table: PerformanceTable, verbose: bool = False) -> str:
    """Render a PerformanceTable as aligned text with per-cell citations."""
    lines: list[str] = []
    lines.append(table.scope_note)
    lines.append(f"\n*** {table.predicate_note} ***\n")

    FIELDS = [
        ("ppa", "PPA (Sensitivity)"),
        ("npa", "NPA (Specificity)"),
        ("lod", "LoD"),
        ("comparator_method", "Comparator/Ref Method"),
        ("predicate_device", "Predicate Device (SE)"),
        ("reactivity_strains", "Reactivity Strains"),
    ]

    for row in table.rows:
        lines.append(f"{'─' * 70}")
        lines.append(f"K-number : {row.k_number}")
        if row.device_name:
            lines.append(f"Device   : {row.device_name}")
        if row.product_code:
            lines.append(f"Prod code: {row.product_code}")
        lines.append("")

        for attr, label in FIELDS:
            pv: Optional[PerformanceValue] = getattr(row, attr)
            if pv is None:
                lines.append(f"  {label:<30} NOT FOUND IN SUMMARY")
            else:
                cit = pv.citation
                page_str = f" p.{cit.page}" if cit.page else ""
                val_display = pv.value.replace("\n", " ")[:120]
                lines.append(f"  {label:<30} {val_display}")
                lines.append(f"  {'':30} ↳ {cit.k_number}{page_str} [{cit.section}]")
                if verbose:
                    lines.append(f"  {'':30}   {cit.source_url}")

        if row.extraction_notes:
            lines.append(f"\n  Notes: {'; '.join(row.extraction_notes)}")
        lines.append("")

    return "\n".join(lines)
