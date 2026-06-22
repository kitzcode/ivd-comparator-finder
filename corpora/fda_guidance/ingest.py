"""
Ingest pipeline for the guidance corpus: fetch -> extract -> section-split ->
store. Mirrors the 510(k) ingest path but emits generic Chunks.

Status values recorded in the manifest:
  "ok"          chunks stored
  "image_only"  PDF had no extractable text (would need OCR)
  "no_pdf"      media id did not serve a fetchable PDF
"""

from __future__ import annotations

from finder.parse.pdf import extract_pdf

from .fetch import fetch_guidance_pdf
from .sections import chunk_guidance
from .seed import GuidanceDoc, SEED_BY_MEDIA_ID
from . import store


def ingest_media_id(media_id: str, *, title: str | None = None) -> dict:
    """Fetch, parse, chunk, and store one guidance document by media id."""
    doc: GuidanceDoc | None = SEED_BY_MEDIA_ID.get(media_id)
    resolved_title = title or (doc.title if doc else f"FDA guidance {media_id}")
    doc_id = doc.doc_id if doc else f"FDA-GUID-{media_id}"
    source_url = doc.source_url if doc else f"https://www.fda.gov/media/{media_id}/download"

    pdf_path = fetch_guidance_pdf(media_id)
    if pdf_path is None:
        store.store_chunks(doc_id, [], status="no_pdf")
        return {"doc_id": doc_id, "status": "no_pdf", "chunks": 0}

    parsed = extract_pdf(pdf_path, doc_id)
    if parsed.is_image_only:
        store.store_chunks(doc_id, [], status="image_only")
        return {"doc_id": doc_id, "status": "image_only", "chunks": 0}

    chunks = chunk_guidance(parsed, doc_id=doc_id, source_url=source_url, title=resolved_title)
    store.store_chunks(doc_id, chunks, status="ok")
    return {"doc_id": doc_id, "status": "ok", "chunks": len(chunks), "pages": parsed.page_count}


def ingest_seed() -> list[dict]:
    """Ingest every document in the curated seed list."""
    from .seed import SEED_GUIDANCES
    return [ingest_media_id(g.media_id) for g in SEED_GUIDANCES]
