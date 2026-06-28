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
_TMP_CHUNK_DIR = Path("/tmp") / "ivd_chunks"

def _pick_writable_dir() -> Path:
    """Return the first chunk dir that is actually writable (not just mkdir-able)."""
    _DEFAULT_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    probe = _DEFAULT_CHUNK_DIR / ".write_probe"
    try:
        probe.write_text("x")
        probe.unlink()
        return _DEFAULT_CHUNK_DIR
    except OSError:
        _TMP_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
        return _TMP_CHUNK_DIR

CHUNK_DIR = _pick_writable_dir()
# Committed chunks live in the project tree even on read-only FS; always check both.
_READ_DIRS = list({_DEFAULT_CHUNK_DIR, CHUNK_DIR})

MANIFEST_PATH = CHUNK_DIR / "_manifest.json"
# Committed manifest is always readable; merge both for status queries.
_COMMITTED_MANIFEST = _DEFAULT_CHUNK_DIR / "_manifest.json"


def _load_manifest() -> dict:
    """Merge committed manifest + writable manifest so all indexed devices are visible."""
    m: dict = {}
    for path in [_COMMITTED_MANIFEST, MANIFEST_PATH]:
        if path.exists():
            try:
                m.update(json.loads(path.read_text()))
            except Exception:
                pass
    return m


def _save_manifest(m: dict) -> None:
    try:
        MANIFEST_PATH.write_text(json.dumps(m, indent=2))
    except OSError:
        pass  # read-only filesystem — skip persisting manifest


def _chunk_path(k_number: str) -> Path:
    from ..security import safe_component
    return CHUNK_DIR / f"{safe_component(k_number)}.json"


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
    """Load stored chunks for a K-number, checking all read dirs. Returns [] if not indexed."""
    from ..security import safe_component
    fname = f"{safe_component(k_number)}.json"
    for d in _READ_DIRS:
        p = d / fname
        if p.exists():
            raw = json.loads(p.read_text())
            return [SummaryChunk(**r) for r in raw]
    return []


def load_chunks_for_product_code(product_code: str) -> list[SummaryChunk]:
    """Load all chunks that belong to a given product code."""
    result: list[SummaryChunk] = []
    seen: set[str] = set()
    for d in _READ_DIRS:
      for p in d.glob("*.json"):
        if p.name.startswith("_") or p.name in seen:
            continue
        seen.add(p.name)
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
