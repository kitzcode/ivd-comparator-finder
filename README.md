# IVD Predicate / Comparator Finder

A tool that maps an IVD assay or analyte to FDA-cleared devices, fetches and parses 510(k) Summary PDFs, and answers grounded performance questions with source citations.

Everything it touches is public FDA data — clean from an IP standpoint and fully reproducible via on-disk caching.

---

## Architecture: one engine, two corpora

The build is split into three reusable layers so the reasoning is independent of any single data source:

```
grounded_rag/      Reasoning layer — corpus-agnostic grounded RAG.
                   Scores chunks, answers from them, refuses on no-match.
                   The model never writes a citation; citations are
                   reconstructed from the chunks it referenced.
      ▲  depends only on the Corpus protocol
corpora/
  fda_510k/        510(k) Summary chunks  ┐  two corpora,
  fda_guidance/    FDA guidance documents ┘  one engine
      ▲  adapts a data source to the protocol (scope + scoring + grounding contract)
finder/            Substrate — openFDA fetch, PDF parse, chunk store/index
```

The grounding contract (refuse on no-match, never show a figure without its source snippet, predicate ≠ comparator) lives once in `grounded_rag`. Each corpus supplies only its data, its scoring weights, and its citation style. The same engine grounds answers over 510(k) Summaries and FDA guidance documents alike.

---

## What it does

| Command | Description |
|---------|-------------|
| `find` | Analyte → device table (K-number, applicant, product code, regulation) |
| `ingest` | Fetch and parse 510(k) Summary PDFs into indexed chunks |
| `ask` | Grounded Q&A over indexed summaries (keyword-only or LLM-backed) |
| `compare` | Structured performance extraction table (PPA, NPA, LoD, comparator, predicate) |
| `status` | Show index status |

## Quick start

```bash
pip install pydantic httpx pdfplumber mcp

# v1: device table
python cli.py find "Group A Strep"

# v2: fetch + parse summaries, then ask a question
python cli.py ingest --knumbers K173653 K141757 K201269
python cli.py ask "What LoD did K173653 report?" --knumbers K173653

# v3: performance comparison table
python cli.py compare --knumbers K173653 K141757 K201269
```

## Web app

```bash
pip install -r requirements.txt
uvicorn app:app --reload      # then open http://localhost:8000
```

Batch-select devices in the left panel to compare them side by side, then
export the metrics table as CSV or JSON. Set `ANTHROPIC_API_KEY` (and optionally
`ANTHROPIC_MODEL`, default `claude-sonnet-4-6`) to enable AI-backed extraction
and Q&A; without it, the app falls back to regex/keyword mode.

## Demo output

```
python cli.py ask "What LoD and reactivity did K173653 report?" --knumbers K173653

Alere™ i Strep A 2 limit of detection (LOD or C95)...
ATCC 12344  147 cells/mL  100%
ATCC 19615   25 cells/mL   95%

REACTIVITY TESTING
ATCC8135, ATCC12384, ATCC12202, ... (14 strains tested)

Citations:
  K173653 p.7 [Performance Testing]
  https://www.accessdata.fda.gov/cdrh_docs/pdf17/K173653.pdf
```

## MCP servers

Two layered, read-only MCP servers map to the architecture:

```
python -m mcp_servers.openfda_device   # DATA layer  — stdio transport
python -m mcp_servers.grounded_rag      # REASONING layer — stdio transport
python -m ivd_mcp                        # deprecated alias for the device server
```

- **openfda_device** (data): `find_devices`, `get_clearance`
- **grounded_rag** (reasoning): `list_corpora`, `ask`, `compare_performance`

`ask` is corpus-parameterized — `corpus="fda_510k"` (scope by `k_numbers` / `product_codes`, cited by K-number) or `corpus="fda_guidance"` (scope by `doc_ids`, cited by guidance tag). All tools are annotated `readOnlyHint=True, destructiveHint=False`.

## Grounding contract

- Every K-number, product code, and regulation comes from a retrieved openFDA record.
- Every performance figure is extracted from an indexed 510(k) Summary PDF, cited by K-number and page.
- Missing data is reported as absent, not invented.
- **Predicate ≠ comparator**: the predicate device (substantial equivalence) and the reference/comparator method (performance study) are always kept distinct.
- Reference-lab results are labeled as directory lookups, not FDA determinations.

## Data sources

- **openFDA** `/device/classification.json` and `/device/510k.json` — device classification and 510(k) clearances
- **510(k) Summary PDFs** from `accessdata.fda.gov` — parsed with `pdfplumber`
- **ARUP Laboratories** and **Mayo Clinic Laboratories** test directories (robots.txt confirmed; ToS-gated; labeled as directory lookups)

## Analyte resolution

openFDA has no clean "analyte" field. Resolution is heuristic:
1. Synonym set (built-in + caller-supplied)
2. Text search across classification and 510(k) device names
3. Supplemental curated product codes for multiplex panels whose device names don't mention the analyte

The synonym set and product code mapping are always surfaced so the user can verify completeness.

## Project structure

```
grounded_rag/        Reasoning layer (corpus-agnostic)
  models.py          generic Chunk / Citation / Answer
  retrieve.py        structural scorer + per-corpus RetrievalConfig
  contract.py        GroundingContract (system prompt, refusal sentinel, citation pattern)
  qa.py              refusal gate, keyword fallback, citation reconstruction
  corpus.py          the Corpus protocol
corpora/
  registry.py        name → Corpus
  fda_510k/          510(k) corpus adapter (grounding + scope→Chunk)
  fda_guidance/      guidance corpus adapter (fetch, section split, ingest)
finder/              Substrate
  models.py          FDA pydantic models
  analyte.py         analyte → product codes (synonym search)
  pipeline.py        v1 find_devices, v2 ingest_summaries
  qa.py              FDA Q&A shim over grounded_rag
  extract.py         structured performance extraction
  sources/           openFDA client + Summary PDF fetch
  parse/             pdfplumber extraction + 510(k) section splitter
  index/             chunk store + keyword retrieval
mcp_servers/
  openfda_device/    data-layer MCP server
  grounded_rag/      reasoning-layer MCP server
ivd_mcp/             deprecated alias for the device server
cli.py
data/cache/          openFDA JSON cache (committed); PDF/chunk cache (gitignored)
tests/               99 tests, incl. a cross-corpus contract test (one engine, two corpora)
```

## Tests

```bash
pytest tests/
# Live lab tests (hits ARUP + Mayo):
pytest tests/test_m5.py --run-live
```

## Milestones

- ✅ M0 — skeleton, models, openFDA client
- ✅ M1 — analyte → device table (v1 CLI)
- ✅ M2 — PDF fetch + parse + chunk store
- ✅ M3 — grounded Q&A with citations
- ✅ M4 — structured performance extraction table
- ✅ M5 — reference-lab directory lookup
- ✅ M6 — read-only MCP server
- ✅ M7 — refactor into three layers: corpus-agnostic `grounded_rag` engine, pluggable `corpora/` (510(k) + FDA guidance), and two layered MCP servers
