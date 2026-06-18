"""
Chunk store: persist and retrieve SummaryChunk objects.

Storage format: one JSON file per K-number under data/cache/chunks/.
A manifest file tracks which K-numbers have been indexed and whether
the source PDF was image-only (and thus skipped).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..models import SummaryChunk

_DEFAULT_CHUNK_DIR = Path(__file__).parent.parent.parent / "data" / "cache" / "chunks"
try:
    _DEFAULT_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_DIR = _DEFAULT_CHUNK_DIR
except OSError:
    CHUNK_DIR = Path("/tmp") / "ivd_chunks"
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = CHUNK_DIR / "_manifest.json"


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(m: dict) -> None:
    try:
        MANIFEST_PATH.write_text(json.dumps(m, indent=2))
    except OSError:
        pass  # read-only filesystem — skip persisting manifest


def _chunk_path(k_number: str) -> Path:
    return CHUNK_DIR / f"{k_number}.json"


def is_indexed(k_number: str) -> bool:
    return _load_manifest().get(k_number) is not None


def get_index_status(k_number: str) -> Optional[str]:
    """Return 'ok', 'image_only', 'no_summary', or None (not indexed)."""
    return _load_manifest().get(k_number)


def store_chunks(k_number: str, chunks: list[SummaryChunk], status: str = "ok") -> None:
    """Persist chunks and record status in the manifest."""
    try:
        _chunk_path(k_number).write_text(
            json.dumps([c.model_dump(mode="json") for c in chunks], indent=2)
        )
    except OSError:
        pass  # read-only filesystem — skip persisting chunks
    m = _load_manifest()
    m[k_number] = status
    _save_manifest(m)


def load_chunks(k_number: str) -> list[SummaryChunk]:
    """Load stored chunks for a K-number. Returns [] if not indexed."""
    p = _chunk_path(k_number)
    if not p.exists():
        return []
    raw = json.loads(p.read_text())
    return [SummaryChunk(**r) for r in raw]


def load_chunks_for_product_code(product_code: str) -> list[SummaryChunk]:
    """Load all chunks that belong to a given product code."""
    result: list[SummaryChunk] = []
    for p in CHUNK_DIR.glob("*.json"):
        if p.name.startswith("_"):
            continue
        try:
            for rec in json.loads(p.read_text()):
                if rec.get("product_code") == product_code:
                    result.append(SummaryChunk(**rec))
        except Exception:
            continue
    return result


def list_indexed() -> dict[str, str]:
    """Return the full manifest: {k_number: status}."""
    return _load_manifest()
