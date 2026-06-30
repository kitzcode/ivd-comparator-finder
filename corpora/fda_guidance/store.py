"""
On-disk store for indexed guidance Chunks: one JSON file per doc_id under
data/cache/guidance_chunks/, plus a manifest of indexed doc_ids and status.

Same shape as the finder chunk store, but it persists the engine's generic
Chunk type (not SummaryChunk). Committed snapshots stay readable on read-only
filesystems; writes fall back to /tmp.
"""

from __future__ import annotations

import json
from pathlib import Path

from grounded_rag.models import Chunk

_DEFAULT_DIR = Path(__file__).parent.parent.parent / "data" / "cache" / "guidance_chunks"
_TMP_DIR = Path("/tmp") / "ivd_guidance_chunks"


def _pick_writable_dir() -> Path:
    _DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
    probe = _DEFAULT_DIR / ".write_probe"
    try:
        probe.write_text("x")
        probe.unlink()
        return _DEFAULT_DIR
    except OSError:
        _TMP_DIR.mkdir(parents=True, exist_ok=True)
        return _TMP_DIR


CHUNK_DIR = _pick_writable_dir()
_READ_DIRS = list({_DEFAULT_DIR, CHUNK_DIR})
MANIFEST_PATH = CHUNK_DIR / "_manifest.json"
_COMMITTED_MANIFEST = _DEFAULT_DIR / "_manifest.json"


def _load_manifest() -> dict:
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
        pass


def _safe_name(doc_id: str) -> str:
    """Sanitize a doc_id into a single filename component so a hostile value
    (e.g. '../../etc/x') can never traverse out of the store directory. Mirrors
    the 510(k) chunk store; matches the invariant documented in finder/security.py."""
    from finder.security import safe_component
    return safe_component(doc_id)


def store_chunks(doc_id: str, chunks: list[Chunk], status: str = "ok") -> None:
    try:
        (CHUNK_DIR / f"{_safe_name(doc_id)}.json").write_text(
            json.dumps([c.model_dump(mode="json") for c in chunks], indent=2)
        )
    except OSError:
        pass
    m = _load_manifest()
    m[doc_id] = status
    _save_manifest(m)


def load_chunks(doc_id: str) -> list[Chunk]:
    fname = f"{_safe_name(doc_id)}.json"
    for d in _READ_DIRS:
        p = d / fname
        if p.exists():
            return [Chunk(**r) for r in json.loads(p.read_text())]
    return []


def load_all_chunks() -> list[Chunk]:
    result: list[Chunk] = []
    seen: set[str] = set()
    for d in _READ_DIRS:
        for p in d.glob("*.json"):
            if p.name.startswith("_") or p.name in seen:
                continue
            seen.add(p.name)
            try:
                result.extend(Chunk(**r) for r in json.loads(p.read_text()))
            except Exception:
                continue
    return result


def get_index_status(doc_id: str):
    return _load_manifest().get(doc_id)


def list_indexed() -> dict[str, str]:
    return _load_manifest()
