# CONTRACTS.md — single source of truth for the Finder refactor

Status: this repo already contains the three-layer refactor (PR #1, 99 tests green).
This document reconciles that build with the formal build brief and pins the
**target** interfaces every module builds against. Where the current code already
satisfies a contract it is marked MET. Where the brief is stricter it is marked
CHANGE, with the exact delta.

No em dashes. Real facts only. Public FDA data only.

---

## 1. Shared types

### Chunk (MET) — `grounded_rag/models.py`
A retrieved passage, corpus-agnostic.
```
Chunk(doc_id, source_url, section, text, page=None, label=None, metadata={})
```
`doc_id` is the stable id within the corpus (510(k) K-number, guidance tag). The
engine never interprets `metadata`.

### SourceRef / Citation (CHANGE: add snippet) — `grounded_rag/models.py`
The brief's `SourceRef` is this build's `Citation`, with one addition. Today:
```
Citation(doc_id, source_url, page=None, section=None, label=None)
```
Target adds the supporting text so the rule "never show a number without its
source snippet" is structural, not implied:
```
Citation(doc_id, source_url, page=None, section=None, label=None, snippet=None)
```

### Answer / AnswerOrRefusal (MET) — `grounded_rag/models.py`
```
Answer(question, answer, citations=[], not_found_reason=None)
```
A populated `not_found_reason` with empty `answer` IS the refusal object.

---

## 2. Corpus protocol — `grounded_rag/corpus.py`

Current (MET, different shape): a corpus exposes `candidates(scope)` plus a
`retrieval_config`, and the engine ranks via `grounded_rag.retrieve.rank`.
`grounded_rag.qa.ask_corpus` composes candidates + rank + answer end to end, which
is exactly the brief's `retrieve(query, filters) -> list[Chunk]` followed by
`answer`.

CHANGE (additive, no break): add a convenience method so the protocol reads as the
brief specifies, implemented in terms of the existing pieces:
```
Corpus.retrieve(query, filters=None, top_k=8, sections=None) -> list[Chunk]
    = rank(query, self.candidates(filters), self.retrieval_config, top_k, sections)
```
`candidates` + `retrieval_config` + `grounding` stay. Filters are corpus-scoped:
`fda_510k` reads `k_numbers` / `product_codes`; `fda_guidance` reads `doc_ids`.

---

## 3. Grounded answerer — `grounded_rag/qa.py`  (THE load-bearing change)

`answer(query, chunks, *, contract, config, llm=None) -> Answer`

### Keyword mode, llm=None (MET)
Returns the most relevant passage of the top chunk verbatim, with a code-built
Citation (now including `snippet`). The model is never invoked. No leakage possible.
This is what the MCP `ask` tool and the default CLI/web path use.

### LLM mode (CHANGE: index-based selection)
Current behavior: candidates are shown to the model with their raw `doc_id`, the
model writes prose containing the identifier, and `_cited_chunks` matches that
emitted id against the retrieved set. The citation list is validated, but the model
emits a real K-number in the visible answer. The brief forbids this.

Target behavior:
1. Code numbers the retrieved candidates `[1]..[N]` and shows the model the text
   and the index only. The model is instructed to cite claims as `[n]`, and NOT to
   write any K-number, URL, accession, or document tag.
2. The model returns: its answer prose using `[n]` markers, plus an explicit list
   of supporting indices (parsed from a structured tail or a fenced block).
3. Code maps each index back to its Chunk and attaches the real `SourceRef`
   (`doc_id`, `source_url`, `page`, `snippet`). Code, not the model, writes every id.
4. Leakage guard: if the model's prose still contains a string matching the
   corpus `cited_id_pattern` or a URL, that is a contract violation. Code blanks
   the offending answer and returns a refusal rather than surfacing it.
5. Indices the model returns that are out of range are dropped (cannot fabricate a
   source that was not retrieved).
6. If the model selects no supporting index, or none clears retrieval, refuse.

### Accepted limitation (honest scope of the guarantee)
The contract guarantees citation PROVENANCE: every cited identifier is real, was in
the retrieved set, and is attached by code, never written by the model. It does NOT
guarantee claim FAITHFULNESS: the engine does not verify by NLI/entailment that a
sentence's prose is semantically supported by the chunk whose index it cites. A
model could cite a valid retrieved chunk for a claim that chunk does not actually
make. Mitigations in place: the leak guard runs on the FULL raw model output (an id
hidden in the SUPPORTING line cannot bypass it); indices are ASCII-only and range
-checked; and keyword mode (the default, and the MCP `ask` path) returns source text
verbatim, which IS fully faithful because no generation occurs. Closing the
faithfulness gap would require a separate verifier model and is out of scope here.

`GroundingContract` (`grounded_rag/contract.py`) keeps `system_prompt`,
`not_found_sentinel`, `cited_id_pattern` (now used as the leakage detector, not the
citation extractor), and the context/user templates (templates change to show
indices, not ids).

---

## 4. Data layer MCP — `mcp_servers/openfda_device/`  (MET, verify fields in RECON)

Read-only tools, `readOnlyHint=True, destructiveHint=False`. Never returns an id
that did not come from openFDA; empty results say so.
- `find_devices(analyte, extra_synonyms?, resolve_summary_urls?) -> {...}`
- `get_clearance(k_number) -> {...} | {error}`

RECON must confirm openFDA field names (`k_number`, `product_code`, `decision_date`,
`applicant_name`, `decision_code`, `clearance_type`) against the live
`api.fda.gov/device/510k.json` and `/device/classification.json` shapes before this
is considered verified.

## 5. Reasoning MCP — `mcp_servers/grounded_rag/`  (MET, inherits §3 change)

- `list_corpora() -> {corpora: {name: description}}`
- `ask(question, corpus="fda_510k", k_numbers?, product_codes?, doc_ids?, top_k=5) -> {...}`
  runs keyword mode (llm=None), so it is leakage-free today and stays so.
- `compare_performance(k_numbers) -> {...}` 510(k) structured extraction; every
  value carries a Citation with the K-number it came from.

---

## 6. Corpora — `corpora/`  (MET)

- `FDA510kCorpus`: analyte-first behavior preserved. `SourceRef` = real K-number +
  accessdata permalink. Retrieval is keyword + section-boost (NO embeddings; the
  reference defers them, RECON confirms).
- `FDAGuidanceCorpus`: same protocol over public guidance PDFs fetched from
  `fda.gov/media/{id}/download`. `SourceRef` = guidance tag + title + public URL.
- `corpora/registry.py`: name -> Corpus. Adding a third corpus is one line.

---

## 7. Regression baseline (MET)

The reference build's original tests (M0-M6) plus the refactor suites, 99 tests,
all passing, are the regression bar. The analyte-first 510(k) queries (eval Q1-Q10
in `tests/test_m6.py`) must keep returning the same grounded, cited answers.

## 8. New tests this refactor adds (to build in Phase 2)

- citation-leakage: an LLM stub that tries to emit a K-number/URL in prose; the
  engine must blank-and-refuse, and the visible answer must contain no raw id.
- source-existence: every Citation.doc_id is present in the retrieved candidate set.
- index-selection: out-of-range indices from the model are dropped, not surfaced.
- one-engine-two-corpora: a single test that runs the unchanged engine over both
  corpora (already present as `tests/test_cross_corpus_contract.py`, extended for
  the index-based path).

---

## 9. Module ownership for Phase 2 (subagents)

| Subagent | Owns | Touches |
|---|---|---|
| rag-core | `grounded_rag/qa.py`, `models.py`, `contract.py`, `corpus.py` | §3 index-based change, §1 snippet |
| data-layer | `mcp_servers/openfda_device/` | field verification only (already built) |
| corpora | `corpora/fda_510k/`, `corpora/fda_guidance/` | §2 retrieve(), §1 snippet wiring |
| tests-evals | `tests/test_*` | §8, against this doc |

A subagent that finds this doc wrong reports back; the main thread fixes
CONTRACTS.md and re-notifies. Subagents do not edit other modules.
