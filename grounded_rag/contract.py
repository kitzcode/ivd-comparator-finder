"""
The grounding contract: everything corpus-specific about *how to answer*,
separated from the generic answering machinery in qa.py.

The anti-hallucination guarantees live here as data, not prose:
  - system_prompt: the rules the LLM is held to (answer only from context, cite
    every figure, refuse when unsupported). Corpus-specific framing, generic spirit.
  - not_found_sentinel: the exact phrase the model is told to emit on refusal;
    qa.py detects it to set not_found_reason.
  - cited_id_pattern: how to find which document ids the model referenced in its
    answer (e.g. r"K\\d{6}" for 510(k)s). If None, qa.py treats a chunk as cited
    when its doc_id appears literally in the answer text. The model never writes a
    Citation object — citations are always reconstructed from retrieved chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

DEFAULT_CONTEXT_TEMPLATE = """\
--- CHUNK {i} | id: {doc_id} | Section: {section} | Page: {page} ---
{text}
"""

DEFAULT_USER_TEMPLATE = """\
Context chunks:

{context}

Question: {question}

Answer (cite the document id and page for every figure):
"""


@dataclass(frozen=True)
class GroundingContract:
    system_prompt: str
    not_found_sentinel: str
    cited_id_pattern: Optional[str] = None
    context_template: str = DEFAULT_CONTEXT_TEMPLATE
    user_template: str = DEFAULT_USER_TEMPLATE
