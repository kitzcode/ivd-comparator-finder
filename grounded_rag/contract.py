"""
The grounding contract: everything corpus-specific about *how to answer*,
separated from the generic answering machinery in qa.py.

The anti-hallucination guarantees live here as data, not prose:
  - system_prompt: the rules the LLM is held to. The model answers ONLY from the
    numbered candidates it is given, cites by bracketed index [n], and NEVER writes
    an identifier (K-number, document tag, accession) or a URL. Code attaches the
    real identifiers afterward, keyed by the indices the model returned.
  - not_found_sentinel: the exact phrase the model is told to emit on refusal;
    qa.py detects it to set not_found_reason.
  - id_leak_pattern: a regex matching identifiers this corpus must never see in the
    model's prose (e.g. r"K\\d{6}" for 510(k)s). qa.py uses it as a LEAKAGE DETECTOR:
    if the model's answer contains a match, the answer is blanked and refused. It is
    NOT used to extract citations. Citations come only from the model's index list.

The context the model sees shows each candidate's index, section, and page, but NOT
its doc_id or source_url, so the model cannot copy an identifier even if it tries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Candidates are shown by INDEX only. No doc_id, no source_url reaches the model.
DEFAULT_CONTEXT_TEMPLATE = """\
[{i}] (Section: {section} | Page: {page})
{text}
"""

DEFAULT_USER_TEMPLATE = """\
Numbered context candidates:

{context}

Question: {question}

Write your answer using bracketed citations like [1], [2] that refer to the
candidates above. Do not write any identifier or URL. End with a final line:
SUPPORTING: [comma-separated indices you relied on]
"""


@dataclass(frozen=True)
class GroundingContract:
    system_prompt: str
    not_found_sentinel: str
    # Regex of identifiers that must never appear in the model's answer (leak guard).
    id_leak_pattern: Optional[str] = None
    context_template: str = DEFAULT_CONTEXT_TEMPLATE
    user_template: str = DEFAULT_USER_TEMPLATE
