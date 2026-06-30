# Build record — IVD Finder three-layer refactor

A factual record of this build for provenance. Capability showcase, not a product.

## What

Refactored a working IVD Predicate/Comparator Finder into three reusable layers
plus a generalization proof:
1. openFDA Device MCP (data layer): typed, read-only tools over openFDA device data.
2. Corpus-agnostic grounded-RAG core (reasoning layer): retrieval plus grounded
   answering with an index-based anti-hallucination contract and a refusal gate.
3. A second corpus, public FDA guidance documents, served by the same core.

## When

- Initial three-layer refactor: 2026-06-22.
- Contract hardening to index-based selection, adversarial suite, and the openFDA
  field-mapping fix: 2026-06-28.

## Where / environment

- Local macOS (Darwin), Python 3.14, pytest 9.
- Dependencies: pydantic v2, httpx, pdfplumber, the official `mcp` Python SDK
  (FastMCP). Keyword retrieval with section-boost scoring; no embeddings, no vector
  database.
- Repository: github.com/kitzcode/ivd-comparator-finder.

## Data, public only

- openFDA device endpoints: `api.fda.gov/device/510k.json`,
  `/device/classification.json`, `/device/pma.json`. US government work, public.
- 510(k) decision summaries: public PDFs on accessdata.fda.gov, fetched per record.
- FDA guidance documents: public PDFs at `fda.gov/media/{id}/download`. The guidance
  search index is bot-walled and was not accessed or bypassed; discovery uses a
  curated seed list of media ids.
- No paywalled standards (ISO, CLSI) are embedded. No PHI. No nonpublic data.

## Verification (verify before trust)

- openFDA field names were verified against cached responses (see RECON.md). This
  surfaced and fixed a real defect: the 510(k) normalizer read `applicant_name` and
  `decision_date_as_string`, which do not exist; the correct fields are `applicant`
  and `decision_date`. The prior code reported the submission date as the decision
  date and an empty applicant.

## Tests

- 121 tests pass, including the original reference suite (regression), a synthetic
  non-FDA corpus exercising the engine, a real ingested FDA guidance document, a
  cross-corpus contract suite, and an adversarial suite (citation leakage,
  source-existence, index-range) run on both corpora.

## Relation to employment

Built from public FDA data only, independent of any employer system, data, or
business. No write-back to any external system. No solicitation or commercial
framing.
