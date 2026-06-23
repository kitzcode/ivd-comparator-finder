"""
MCP servers exposing the three layers as typed, read-only tools.

  - openfda_device — the DATA layer: FDA device facts from openFDA
    (find_devices, get_clearance). No RAG, no PDFs.
  - grounded_rag   — the REASONING layer: grounded Q&A and performance
    extraction over registered corpora (510(k) and guidance).
"""
