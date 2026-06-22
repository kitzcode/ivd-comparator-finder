"""
FDA guidance-document corpus adapter.

The second corpus on the grounded_rag engine, proving the reasoning layer is
source-agnostic: one engine, two corpora. Guidance PDFs are fetched from
fda.gov/media/{id}/download (the search index is bot-walled and not used), parsed
through the shared PDF pipeline, section-split on guidance structure, and indexed
as generic Chunks.
"""

from __future__ import annotations

from .corpus import FDAGuidanceCorpus, FDA_GUIDANCE_CONTRACT, FDA_GUIDANCE_RETRIEVAL
from .seed import SEED_GUIDANCES, GuidanceDoc
from .ingest import ingest_media_id, ingest_seed

__all__ = [
    "FDAGuidanceCorpus",
    "FDA_GUIDANCE_CONTRACT",
    "FDA_GUIDANCE_RETRIEVAL",
    "SEED_GUIDANCES",
    "GuidanceDoc",
    "ingest_media_id",
    "ingest_seed",
]
