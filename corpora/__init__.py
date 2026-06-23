"""
Corpus adapters: concrete sources plugged into the grounded_rag engine.

Each subpackage implements the grounded_rag.Corpus protocol over a real body of
documents:
  - fda_510k    — FDA 510(k) decision-summary chunks (the original build).
  - fda_guidance — FDA guidance documents (proves the engine generalizes).
"""
