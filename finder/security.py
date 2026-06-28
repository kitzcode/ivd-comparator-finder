"""
Input validation / sanitization shared across the engine and web layer.

Defense in depth: the web layer validates identifiers at the boundary, and the
filesystem/cache sinks sanitize again so a hostile value can never escape the
cache directory even if a new caller forgets to validate.
"""

from __future__ import annotations

import re

# FDA submission identifiers: 1–3 uppercase letters + 6 digits.
# Covers K173653 (510(k)), DEN140005 (De Novo), P160030 (PMA), BK251286 (CBER).
_DEVICE_ID = re.compile(r"^[A-Z]{1,3}[0-9]{6}$")

# Characters allowed in a cache filename component. Anything else (including
# '/', '\\', '.', whitespace) is replaced — kills path traversal at the sink.
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")

# Per-request work caps to bound cost and resource use on unauthenticated calls.
MAX_IDS_PER_REQUEST = 25
MAX_ANALYTE_LEN = 120


def is_valid_device_id(value: str) -> bool:
    """True if value is a well-formed FDA submission number."""
    return bool(_DEVICE_ID.match(value or ""))


def safe_component(value: str, max_len: int = 80) -> str:
    """
    Make a string safe to use as a single filename component.
    Replaces any non-[A-Za-z0-9_-] character and caps length, so values like
    '../../etc/passwd' collapse to '______etc_passwd' and cannot traverse.
    """
    cleaned = _UNSAFE.sub("_", value or "")
    return cleaned[:max_len] or "_"


def escape_query(value: str) -> str:
    """Escape a value for safe interpolation inside an openFDA search string."""
    return (value or "").replace("\\", "\\\\").replace('"', '\\"')
