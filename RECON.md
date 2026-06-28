# Phase 1 Recon — IVD Predicate/Comparator Finder

Scope: verify, do not assume. Evidence is drawn from (a) live openFDA docs via web
search, (b) committed cache JSON under `data/cache/`, and (c) the repo's own code.
Where a live check was blocked by the sandbox, that is stated explicitly rather than
guessed. Public FDA data only.

Date of recon: 2026-06-27.

---

## 1. openFDA device endpoints and exact field names — PARTIAL (one real defect found)

### Endpoints — CONFIRMED
- 510(k): `https://api.fda.gov/device/510k.json` — `finder/sources/openfda.py:28`
  (`FIVEK_EP`). Base is `https://api.fda.gov` (`openfda.py:26`).
- Classification: `https://api.fda.gov/device/classification.json` —
  `finder/sources/openfda.py:27` (`CLASSIFICATION_EP`).
- PMA (also used): `https://api.fda.gov/device/pma.json` — `openfda.py:29`.

### 510(k) field names actually present (cached record `data/cache/fivek_k_K173653.json`)
Top-level fields in a real result object:
- `k_number` (e.g. `"K173653"`) — line 75
- `product_code` (`"PGX"`) — line 21
- `decision_date` (`"2018-05-02"`, ISO `YYYY-MM-DD`) — line 68
- `applicant` (`"Alere Scarborough, Inc."`) — line 67  **NOTE: it is `applicant`, NOT `applicant_name`**
- `device_name` — line 71
- `decision_code` (`"SESE"`) — line 69
- `decision_description` (`"Substantially Equivalent"`) — line 80
- `advisory_committee` (`"MI"`) — line 72
- `advisory_committee_description` (`"Microbiology"`) — line 18
- `review_advisory_committee` (`"MI"`) — line 78
- `statement_or_summary` (`"Summary"`) — line 20
- `date_received` (`"2017-11-28"`) — line 77
- `clearance_type` (`"Dual Track"`) — line 81
- `third_party_flag`, `expedited_review_flag`, address/city/state/zip fields also present.
- Nested `openfda` object carries: `device_name`, `regulation_number` (`"866.2680"`,
  line 63), `device_class` (`"2"`), `medical_specialty_description`, plus
  registration/FEI number arrays. **`regulation_number` lives under `openfda`, not at top level, in 510(k) records.**

Live docs cross-check (open.fda.gov 510(k) searchable-fields, via web search):
confirms the field is `applicant` (it describes pre/post Aug-14-2014 semantics of
`applicant`), and confirms `decision_date`, `k_number` as real fields. No mention of
an `applicant_name` or `decision_date_as_string` field on the 510(k) endpoint.

### Classification field names (cached record `data/cache/cls_pc_GTY.json`)
- `product_code` (`"GTY"`) — line 19
- `device_name` (`"Antigens, All Groups, Streptococcus Spp."`) — line 178
- `regulation_number` (`"866.3740"`, top level here) — line 184
- `device_class` (`"1"`) — line 182
- `medical_specialty` (`"MI"`) / `medical_specialty_description` (`"Microbiology"`) — lines 177, 181
- `review_panel` (`"MI"`) — line 176
- `definition` (empty string here) — line 183
- `submission_type_id`, `implant_flag`, several `*_flag` fields, and a nested
  `openfda` object holding `k_number` and registration-number arrays.

### How the code reads these fields
- `finder/sources/openfda.py` only ever reads `results` / `meta.results.total` from
  responses and passes records through untouched — its search queries use the correct
  field names (`product_code`, `device_name`, `definition`, `k_number`, `pma_number`).
  CONFIRMED correct.
- `finder/pipeline.py::_normalize_device` (510(k) path) reads:
  - `rec.get("applicant_name", "")` (`pipeline.py:43`) — **field does not exist in the
    510(k) response; the real field is `applicant`.** Result: `applicant_name` is
    silently empty for every 510(k)/De Novo device.
  - `rec.get("decision_date_as_string") or rec.get("date_received")` (`pipeline.py:44`)
    — **`decision_date_as_string` does not exist in the 510(k) response.** The real
    field `decision_date` is never read; the code falls back to `date_received`
    (the submission-received date, not the clearance/decision date). So `decision_date`
    on the Device is the wrong date for 510(k) records.
  - `rec.get("traditional_501k_flag")` is mapped to `predicate_k_number`
    (`pipeline.py:49`) — that flag is not a predicate K-number; it is unverified and
    looks semantically wrong, but it is out of the strict scope of this item.
- `_normalize_pma` (PMA path) correctly reads `applicant`, `decision_date`,
  `pma_number`, `trade_name`/`generic_name` (`pipeline.py:54-65`). CONFIRMED correct
  for PMA.
- The MCP `find_devices` / `get_clearance` tools (`mcp_servers/openfda_device/server.py`)
  surface `rec.get("applicant_name")` and the Device's `applicant_name` field, so the
  empty-applicant defect propagates to MCP output.

Verdict: endpoints and the true field names are CONFIRMED. Field *consumption* in
`pipeline.py` is NOT fully correct: two fields it reads for 510(k) records
(`applicant_name`, `decision_date_as_string`) do not exist in openFDA responses.

---

## 2. 510(k) decision summaries: sourcing and parsing — CONFIRMED

- Summaries are fetched from **accessdata.fda.gov**, not from openFDA full text.
  `finder/sources/summaries.py:34-38` defines the 510(k) Summary URL patterns:
  - `https://www.accessdata.fda.gov/cdrh_docs/pdf{yy}/{k}.pdf`
  - `https://www.accessdata.fda.gov/cdrh_docs/reviews/{k}.pdf`
  - `https://www.accessdata.fda.gov/cdrh_docs/pdf/{k}.pdf`
  `{yy}` is the 2-digit year derived from the K-number digits (`_year2`, lines 57-59).
- PMA SSED ("B" document) patterns at `summaries.py:42-46`:
  - `…/cdrh_docs/pdf{yy}/{k}B.pdf`, `…/cdrh_docs/pdf/{kl}b.pdf`, `…/cdrh_docs/pdf{yy}/{k}b.pdf`.
  `_patterns_for` (lines 49-51) routes P-numbers to the SSED patterns, everything else
  to the 510(k) patterns.
- openFDA does **not** return summary text. The 510(k) record exposes only
  `statement_or_summary` = `"Summary"` / `"Statement"` (a flag indicating which exists),
  confirmed in `data/cache/fivek_k_K173653.json:20`. There is no full-text-of-summary
  field. The code never tries to read summary text from openFDA; it always probes
  accessdata.
- URL probing technique (matches CLAUDE.md note): `GET` with `Range: bytes=0-3`,
  accepting status 200/206 with a `pdf` content-type, or a 200 whose first bytes are
  `%PDF` (`summaries.py:98-109`). HEAD is deliberately avoided. Resolved URLs are
  cached in a `.url` sidecar per K-number (`summaries.py:66-72, 125-128`); committed
  sidecars are checked first so it works on read-only filesystems.
- Parsing: `finder/parse/pdf.py` uses **pdfplumber** (`import pdfplumber`,
  `pdf.py:71`) to extract per-page `text` (`page.extract_text()`, line 81) and tables
  (`page.extract_tables()`, line 85). Image-only PDFs are flagged when avg chars/page
  < 20 (`pdf.py:100`); `summaries.is_image_only_pdf` provides a pre-check
  (`summaries.py:175-184`).

Verdict: CONFIRMED. Summaries come from accessdata.fda.gov via per-record URL probing;
openFDA carries only a Summary/Statement flag, not the text; pdfplumber does the
extraction.

---

## 3. FDA guidance documents: location and format — CONFIRMED (live byte-probe blocked, otherwise corroborated)

- Direct serving pattern is `https://www.fda.gov/media/{media_id}/download`
  (`corpora/fda_guidance/fetch.py:29-30`, `media_url`; same string in
  `corpora/fda_guidance/seed.py:33-34`, `GuidanceDoc.source_url`).
- The guidance **search index** page
  (`fda.gov/regulatory-information/search-fda-guidance-documents`) is documented in
  `seed.py:6-12` as bot-walled, in the same class as the cfPMN database, and is **not**
  to be bypassed. This recon did not attempt it. Web search independently shows the
  search-index page exists at that URL as a distinct HTML page, separate from the
  media-download path.
- Fetch technique mirrors the accessdata prober: `GET` with `Range: bytes=0-3`,
  accept 200/206 + `pdf` content-type, else 200 + `%PDF` magic bytes
  (`fetch.py:37-47`); full download verifies `content[:4] == b"%PDF"`
  (`fetch.py:59`).
- Live lightweight check on one media id: ATTEMPTED but the sandbox **blocked** both
  `curl` and a Python one-liner (Bash exec denied), so I could not personally observe
  the `content-type: application/pdf` / HTTP 206 header on `media/71075/download`.
  Corroborating evidence instead:
  - Web search resolves `https://www.fda.gov/media/71075/download` to the document
    titled "Guidance for Industry and FDA Staff … In Vitro Diagnostic (IVD) Device
    Studies — Frequently Asked Questions," which exactly matches the seed entry
    `GuidanceDoc("71075", "In Vitro Diagnostic (IVD) Device Studies - Frequently Asked
    Questions", …)` in `seed.py:39`.
  - `seed.py:10-12` records a prior in-repo verification: "verified: HTTP 206,
    content-type application/pdf, born-digital text," and the seed header says the set
    was "verified fetchable 2026-06-22" (`seed.py:37`).

Verdict: CONFIRMED for the direct `media/{id}/download` PDF path and for the
search-index being a separate, bot-walled page that must not be bypassed. The single
live content-type byte-probe could not be run here (sandbox blocked network exec); it
is corroborated by web search + the code's own recorded prior verification, not by a
fresh header observation. See Flags.

---

## 4. Retrieval approach: keyword vs embeddings — CONFIRMED (keyword only, no embeddings)

- `grounded_rag/retrieve.py` is pure keyword scoring. Module docstring states "Generic
  keyword retrieval and scoring. No embeddings, no FDA knowledge" (lines 1-2). Scoring
  is term-frequency with diminishing returns `1 + ln(count)`, per-term domain weights,
  `sqrt(distinct tokens)` length normalization, a coverage bonus, and a per-section
  multiplier (`score_chunk`, lines 70-100). No vector math, no model calls.
- `finder/index/retrieve.py` is the FDA adapter over that engine. It supplies the FDA
  `RetrievalConfig`: section boosts (Performance Testing 2.0, etc., lines 22-27),
  IVD domain-term weights (lod/ppa/npa/loq 3.0, etc., lines 30-37), and IVD stopwords
  (lines 40-42). It calls `grounded_rag.retrieve.rank` (line 101). No embeddings.
- Dependencies: `requirements.txt` lists only `pydantic`, `httpx`, `pdfplumber`,
  `mcp[cli]`, `fastapi`, `uvicorn`, `anthropic`. **No embedding/vector library**
  (no sentence-transformers, faiss, chromadb, numpy-for-vectors, openai-embeddings,
  etc.). CONFIRMED the reference defers embeddings.

Verdict: CONFIRMED. Keyword scoring with section boosts and domain-term weights; no
embeddings; no embedding dependency in `requirements.txt`.

---

## 5. MCP SDK in use — PARTIAL (SDK + imports confirmed; exact installed version not confirmable here)

- Server uses **FastMCP from the official `mcp` Python SDK**:
  `from mcp.server.fastmcp import FastMCP` (`mcp_servers/openfda_device/server.py:25`).
- Tool annotations come from `mcp.types`:
  `from mcp.types import ToolAnnotations` (`server.py:26`), used as
  `ToolAnnotations(readOnlyHint=True, destructiveHint=False)` (`server.py:37`) on every
  tool. CONFIRMED.
- Server instance: `mcp = FastMCP("openfda-device", instructions=…)` (`server.py:28`),
  tools registered with `@mcp.tool(...)`, entrypoint `mcp.run()` (`server.py:139`).
- Dependency pin: `mcp[cli]>=1.0` (`requirements.txt:4`).
- Installed version: NOT CONFIRMED. There is no project-local `venv`/`.venv`
  (checked — none exists), so `mcp` is installed in a global/system environment outside
  the project tree. `pip show mcp` and `python -c importlib.metadata` were both blocked
  by the sandbox (Bash exec denied), and a broad `find /` for the dist-info was also
  blocked. I could not read the installed version. The code only requires `>=1.0`, and
  the imports it uses (`mcp.server.fastmcp.FastMCP`, `mcp.types.ToolAnnotations`) are
  present in mcp SDK 1.x.

Verdict: PARTIAL. SDK identity and import paths CONFIRMED (official `mcp` SDK,
FastMCP + ToolAnnotations). Exact installed version unverified in this environment.

---

## Flags (treat as unverified or as known defects for any build module)

1. **DEFECT — 510(k) `applicant` field misread.** `finder/pipeline.py:43` reads
   `applicant_name`, which does not exist on `/device/510k.json` records (real field:
   `applicant`). Every 510(k)/De Novo Device gets an empty `applicant_name`, and this
   empties out the MCP `find_devices`/`get_clearance` applicant output. PMA path is
   correct. Must be fixed in the refactor.

2. **DEFECT — 510(k) decision date misread.** `finder/pipeline.py:44` reads
   `decision_date_as_string` (nonexistent) then falls back to `date_received`. The real
   `decision_date` field is never used, so the Device's `decision_date` is actually the
   submission-received date for 510(k) records. Confirmed against both the cached record
   and live docs.

3. **UNVERIFIED — predicate mapping.** `pipeline.py:49` maps `traditional_501k_flag`
   to `predicate_k_number`. That flag is not a predicate K-number; the field name is
   also misspelled relative to openFDA conventions. Flagged as semantically wrong /
   unverified; outside item-1 scope but relevant to the "predicate ≠ reference method"
   invariant.

4. **UNVERIFIED LIVE — guidance PDF content-type.** Could not run the fresh
   `Range: bytes=0-7` byte-probe on `media/71075/download` (sandbox blocked curl and
   python exec). The `media/{id}/download` PDF path is corroborated by web search and by
   the code's recorded prior verification (HTTP 206, application/pdf, 2026-06-22), but a
   build module should re-run a live probe before relying on freshness.

5. **UNVERIFIED — installed `mcp` version.** No project venv; version-introspection
   commands were sandbox-blocked. Only the pin `mcp[cli]>=1.0` and the import paths are
   confirmed. Confirm the concrete installed version before assuming any 1.x-specific
   FastMCP behavior.

6. **Note (not a defect) — `regulation_number` location differs by endpoint.** It is
   nested under `openfda.regulation_number` in 510(k) records but top-level in
   classification records. The pipeline sources `regulation_number` from the
   classification/ProductCodeInfo path, so this is handled, but any code reading it
   straight off a raw 510(k) record must use the `openfda` sub-object.
