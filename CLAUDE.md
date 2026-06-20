# CLAUDE.md — IVD Comparator Finder

## What this is

A tool that maps an IVD assay / analyte to FDA-cleared devices via openFDA, fetches 510(k) Summary PDFs, and answers grounded performance questions with citations. All output is traceable to public FDA data.

## Build status

- `finder/models.py` — pydantic v2 models (Device, ProductCodeInfo, AnalyteResolution, SummaryChunk, Citation, Answer). Done.
- `finder/sources/openfda.py` — openFDA classification + 510(k) client with on-disk cache. Done.
- `finder/sources/summaries.py` — resolve + fetch 510(k) Summary PDFs (GET with Range header; accessdata returns 404 on HEAD). Done.
- `finder/analyte.py` — analyte term → product codes via synonym text search + supplemental curated panel codes. Done.
- `finder/parse/pdf.py` — pdfplumber extraction: text + tables per page, image-only detection. Done.
- `finder/parse/sections.py` — heading-pattern section splitter → SummaryChunk list. Done.
- `finder/index/store.py` — chunk store (JSON per K-number) + manifest. Done.
- `finder/index/retrieve.py` — keyword retrieval with section-boost scoring. Done.
- `finder/qa.py` — grounded Q&A: keyword-only mode (no LLM) or LLM-backed with Anthropic SDK. Done.
- `finder/extract.py` — M4 structured performance extraction (PPA, NPA, LoD, reactivity, comparator, predicate) via regex + optional LLM fill. Every value carries a Citation. Done.
- `finder/pipeline.py` — v1 find_devices() + v2 ingest_summaries(). Done.
- `ivd_mcp/ivd_server.py` — M6 FastMCP server: 5 read-only tools, `readOnlyHint=True`. Note: the package is `ivd_mcp/` (not `mcp/`) to avoid shadowing the installed `mcp` package. Done.
- `cli.py` — `find`, `ingest`, `ask`, `compare`, `status` commands. Done.
- `tests/` — 71 tests (M0–M6), all passing. Done.

## Invariants

1. **Grounded or silent.** Every K-number, product code, and performance figure must trace to a retrieved record. Never fill from model memory.
2. **Predicate ≠ reference method.** Keep distinct: the predicate device (substantial equivalence) vs. the reference/comparator method (performance study).
3. **Reproducible.** All raw openFDA fetches are cached to `data/cache/`. Tests run against cache, not live API.
4. **Heuristic analyte resolution is always labeled.** Surface synonyms used and warn the user.
5. **Engine-first.** No write-back to any system. Public data only. No PHI.

## Analyte resolution design

`finder/analyte.py`:
1. Build synonym set from `BUILTIN_SYNONYMS` (+ caller extras).
2. Text-search openFDA classification + 510(k) device names (synonyms ≥ 6 chars to avoid false positives).
3. Add `SUPPLEMENTAL_PRODUCT_CODES` — curated panel product codes whose device names don't mention the analyte (e.g. BioFire BCID2 = `PEN`, 866.3365, for GAS).
4. Fetch canonical classification metadata for each product code.
5. Return `AnalyteResolution` with all product codes + heuristic warning.

**openFDA returns 404 for zero-result queries** — handled gracefully as empty results.

## Data cache

`data/cache/*.json` — one file per API request, keyed by query parameters. Gitignored (large). Commit named snapshots as `data/cache/snapshot_<study>_<date>.json` for test reproducibility.

Set `OPENFDA_API_KEY` env var to raise rate limits (240/min → 240k/day).

## Key implementation note: accessdata PDF URL probing

`accessdata.fda.gov` returns 404 for HEAD requests but 200 for GET. The prober uses `GET` with `Range: bytes=0-3` to avoid downloading the full PDF. The resolved URL is cached in a `.url` sidecar file per K-number.

## Regulatory pathways covered

openFDA's `/device/510k.json` is CDRH-only. Coverage now spans three pathways,
each tagged on `Device.submission_type`:
- **510(k)** — `/device/510k.json` (K-numbers). Full pipeline: list, parse Summary PDF, Q&A, performance.
- **De Novo** — same dataset, DEN-numbered grants (`decision_code` DENG). Detected and labeled; same pipeline.
- **PMA** — `/device/pma.json` (P-numbers). Listed per product code (supplements deduped to one row).
  SSED documents parse through the SAME pipeline: `resolve_summary_url` returns the SSED ("B" doc) URL
  for P-numbers (`cdrh_docs/pdf{yy}/{P}B.pdf`, older `cdrh_docs/pdf/{p}b.pdf`). SSEDs are longer/less
  standardized than 510(k) Summaries, so extraction is best-effort.

**Not covered: CBER 510(k)s (BK-numbers) and BLA biologics.** Many HIV/HCV/HBV blood tests are CBER
products. They are absent from openFDA (any endpoint) AND the CDRH bulk download files. The only source
is the FDA web cfPMN database, which is behind bot-detection ("Challenge Validation") — do NOT attempt to
bypass it. CBER discovery-by-analyte is therefore not automatable here.

## Open milestones
- **M4**: Structured performance extraction (PPA/NPA/LoD/reactivity) into comparison table.
- **M6**: MCP server (`find_devices`, `get_clearance`, `ask_summary`, `compare_performance`). 10 eval questions against cached snapshot.

## Verify-before-asserting

- openFDA field names and query syntax: confirm against current openFDA docs.
- accessdata PDF URL patterns: probe per record; handle missing Summaries/SSEDs.
- PDF table extraction: verify fidelity; detect scanned/image-only PDFs before trusting numbers.
