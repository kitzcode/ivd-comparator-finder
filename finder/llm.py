"""
Shared Anthropic LLM factory for grounded extraction and Q&A.

The engine is model-agnostic: qa.ask() and extract.extract_performance() take a
callable `(system_prompt, user_prompt) -> str`. This module builds that callable
from the Anthropic SDK when an API key is configured, and returns None otherwise
so callers transparently fall back to keyword/regex mode.

Configuration (env vars):
  ANTHROPIC_API_KEY  — required to enable LLM mode. Without it, make_llm()
                       returns None and the app uses keyword/regex extraction.
  ANTHROPIC_MODEL    — optional model override (default: claude-sonnet-4-6).
"""

from __future__ import annotations

import os
from typing import Callable, Optional

_DEFAULT_MODEL = "claude-sonnet-4-6"


def llm_enabled() -> bool:
    """True if an Anthropic API key is configured."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def make_llm(model: Optional[str] = None) -> Optional[Callable[[str, str], str]]:
    """
    Return an `(system_prompt, user_prompt) -> str` callable backed by the
    Anthropic SDK, or None if no API key / SDK is available (caller falls back).
    """
    if not llm_enabled():
        return None
    try:
        import anthropic
    except ImportError:
        return None

    client = anthropic.Anthropic()
    resolved_model = model or os.environ.get("ANTHROPIC_MODEL", _DEFAULT_MODEL)

    def call(system_prompt: str, user_prompt: str) -> str:
        msg = client.messages.create(
            model=resolved_model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return msg.content[0].text

    return call
