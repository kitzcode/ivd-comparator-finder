"""
Curated seed list of FDA guidance documents.

Discovery-by-topic is NOT cleanly automatable: the FDA guidance *search index*
(fda.gov/regulatory-information/search-fda-guidance-documents) is behind
bot-detection, the same class of wall as the cfPMN database. We do NOT bypass it.

Individual guidance PDFs, however, are served openly from
https://www.fda.gov/media/{media_id}/download and are directly fetchable
(verified: HTTP 206, content-type application/pdf, born-digital text). So this
corpus uses a hand-curated seed list of media ids, exactly as the device layer
uses SUPPLEMENTAL_PRODUCT_CODES for panel codes the text search misses.

Each entry's doc_id is "FDA-GUID-{media_id}" — the stable citation tag.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuidanceDoc:
    media_id: str
    title: str
    topic: str

    @property
    def doc_id(self) -> str:
        return f"FDA-GUID-{self.media_id}"

    @property
    def source_url(self) -> str:
        return f"https://www.fda.gov/media/{self.media_id}/download"


# IVD-focused seed set (verified fetchable 2026-06-22). Extend as needed.
SEED_GUIDANCES: list[GuidanceDoc] = [
    GuidanceDoc("71075", "In Vitro Diagnostic (IVD) Device Studies - Frequently Asked Questions", "IVD studies"),
    GuidanceDoc("92930", "Establishing the Performance Characteristics of In Vitro Diagnostic Devices for HPV Detection", "performance characteristics"),
    GuidanceDoc("81309", "In Vitro Companion Diagnostic Devices", "companion diagnostics"),
    GuidanceDoc("87374", "Distribution of In Vitro Diagnostic Products Labeled for Research Use Only or Investigational Use Only", "RUO/IUO labeling"),
    GuidanceDoc("111186", "Replacement Reagent and Instrument Family Policy for In Vitro Diagnostic Devices", "replacement reagent policy"),
]

SEED_BY_DOC_ID: dict[str, GuidanceDoc] = {g.doc_id: g for g in SEED_GUIDANCES}
SEED_BY_MEDIA_ID: dict[str, GuidanceDoc] = {g.media_id: g for g in SEED_GUIDANCES}
