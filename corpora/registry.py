"""
Corpus registry: name -> Corpus, so the reasoning layer (CLI, MCP) can target
any registered corpus without importing concrete adapters directly.

Adding a third corpus is a one-line change here; nothing in grounded_rag changes.
"""

from __future__ import annotations

from typing import Callable

from grounded_rag.corpus import Corpus

# Lazy factories so importing the registry doesn't pull every corpus' deps.
_FACTORIES: dict[str, Callable[[], Corpus]] = {}
_DESCRIPTIONS: dict[str, str] = {}


def _register(name: str, factory: Callable[[], Corpus], description: str) -> None:
    _FACTORIES[name] = factory
    _DESCRIPTIONS[name] = description


def _fda_510k() -> Corpus:
    from corpora.fda_510k import FDA510kCorpus
    return FDA510kCorpus()


def _fda_guidance() -> Corpus:
    from corpora.fda_guidance import FDAGuidanceCorpus
    return FDAGuidanceCorpus()


_register(
    "fda_510k", _fda_510k,
    "FDA 510(k) decision-summary chunks. Scope keys: k_numbers, product_codes. "
    "Citations by K-number + page.",
)
_register(
    "fda_guidance", _fda_guidance,
    "FDA guidance documents. Scope key: doc_ids (FDA-GUID-NNNNN). "
    "Citations by guidance tag + page.",
)


def get_corpus(name: str) -> Corpus:
    if name not in _FACTORIES:
        raise KeyError(f"Unknown corpus {name!r}. Known: {sorted(_FACTORIES)}")
    return _FACTORIES[name]()


def list_corpora() -> dict[str, str]:
    """Return {name: description} for every registered corpus."""
    return dict(_DESCRIPTIONS)
