# EvidenceEngine Architecture

## Overview

EvidenceEngine is a biomedical claim verification pipeline. You upload a manuscript (the *report*) and the studies it cites; the system extracts factual claims, retrieves supporting passages from the cited documents, classifies each claim with an NLI model, and presents the result on a reviewer dashboard with full character-offset provenance.

There are no LLM calls in the verdict hot path. Claim extraction is local NLTK + heuristic filters. Verdict classification is a local cross-encoder NLI model. The only optional external call is the on-demand "Explain this verdict" button, which sends a single claim and its top evidence spans to an LLM for a one-paragraph plain-English explanation. Everything else runs on the server with no API keys required.

The system is deliberately conservative. A claim is `supported` only when the NLI model is highly confident in entailment AND the retrieved evidence is from a real cited source (not just the same paper). Anything below the confidence threshold falls to `needs_review` or `insufficient_support`.

What distinguishes EvidenceEngine from a generic RAG pipeline is the provenance data model. Every evidence span carries the character offsets of the text it came from. Every verdict is linked to the exact spans that informed it via a `VerdictEvidence` join table. A reviewer can click a verdict and see the verbatim sentence from the source document at the verified offset, not a paraphrase.

---

## Pipeline at a glance

```
PDF upload  →  Parse  →  Extract claims  →  Retrieve evidence  →  Classify  →  Reviewer dashboard
                              │                  │                  │
                              ▼                  ▼                  ▼
                         NLTK + filter     BM25 + dense          NLI cross-encoder
                         (no LLM)          + reranker            (no LLM)
```

Five stages, all running locally.

---

## Stages

### 1. Parse

`src/evidenceengine/ingestion/` — Accepts PDF or DOCX. Parses with PyMuPDF, producing a list of text blocks each carrying `char_start` and `char_end` offsets into the document's raw text. The parse-then-accept pattern means a failed parse rolls the whole upload back; partial state never lands in the DB.

Table extraction is skipped for PDFs over 50 pages (PyMuPDF's table heuristic is slow on large documents and the win is small for biomedical reviews).

### 2. Extract claims

`src/evidenceengine/extraction/claim_extractor.py`

NLTK sentence tokenizer plus a stack of filters. Filters reject:

- Bibliography entries (journal-abbreviation endings, author lists with year)
- Methods boilerplate (procedural language without finding verbs)
- Glossary entries and figure captions
- Page headers, license text, journal mastheads
- Sentences starting with `-ing` participles (typically procedural)
- Sentences with too high an `NNP` (proper-noun) ratio (typically affiliation lists)

The filter is `_is_likely_claim()` plus `_LEGAL_UNIVERSAL_RE` (universal legal/structural rejection patterns) and `_FACTUAL_SIGNAL_RE` (positive gate requiring biomedical signal: numeric values with units, study terms like "patients", "cohort", "dose", "p<", "RR=", etc.).

Citation-first sampling biases the 25-claim cap toward sentences with citation markers (Vancouver `[1]`, APA `(Smith et al., 2023)`). The cap (`max_claims_absolute=25`, `max_claims_per_page=4`) keeps the pipeline finishable on a shared CPU. See `CHALLENGES.md` for the trade-off.

The extraction stage is offloaded to a thread pool (`asyncio.to_thread`) so that NLTK's synchronous CPU work does not block the async event loop.

### 3. Retrieve evidence

`src/evidenceengine/retrieval/pipeline.py`

Hybrid retrieval: BM25 (keyword) + BGE-base-en-v1.5 dense embeddings (semantic), unioned and reranked by `cross-encoder/ms-marco-MiniLM-L6-v2`.

- **BM25** (`bm25s` library): per-source-document index, queried with the claim text. Disk-cached at `INDEX_DIR`.
- **Dense** (`BAAI/bge-base-en-v1.5`): SentenceTransformer model, max_seq_length=512, embeddings disk-cached by content hash. Falls back to BM25-only when the corpus exceeds `dense_retrieval_max_corpus_size` (3000 chunks; see `core/config.py`). Fail-open: if the dense path errors at runtime the pipeline still returns BM25 results.
- **Reranker** (`cross-encoder/ms-marco-MiniLM-L6-v2`): scores the union and keeps the top-N. The larger `bge-reranker-v2-m3` is intentionally NOT used — at 568M params it takes 70s per batch on shared CPU.

Top spans become `EvidenceSpan` rows carrying `char_start`, `char_end`, `relevance_score`, and the `source_document_id` they came from.

For a claim with a resolved citation anchor (e.g. `[1]` resolves to a specific cited PDF), retrieval is scoped to that document (Mode B1). When EVERY anchor on a claim is unresolvable (anchor exists, no upload matches), retrieval is skipped for that claim and the verdict is set to `needs_review` (Mode B2). When the claim sentence has no citation anchor at all, retrieval falls back to broad multi-document search across all uploaded cited papers (Mode B3), or to the report itself if there are no cited documents (Mode A self-verify).

### 4. Classify

`src/evidenceengine/classification/pipeline.py` and `nli_classifier.py`

Uses `cross-encoder/nli-deberta-v3-small` fine-tuned on SciFact (the biomedical claim verification dataset). The fine-tuned copy is not in this repo: point `NLI_MODEL_PATH` at one, or the loader uses the unmodified base model from Hugging Face.

For each claim, the classifier scores claim-vs-each-evidence-span pairs in NLI form (premise = evidence, hypothesis = claim). The model produces three probabilities: `entailment`, `contradiction`, `neutral`. Per-span scores are aggregated into a single verdict via `nli_probs_to_verdict()` in `nli_classifier.py`.

Threshold band (locked by `test_config.py` invariants):

```
nli_entailment_supported_threshold        = 0.92   # entailment must clear this for supported
nli_contradiction_contradicted_threshold  = 0.75   # contradiction must clear this for contradicted
nli_tiebreaker_threshold                  = 0.65   # reserved for future tiebreaker path
nli_min_confidence_for_verdict            = 0.50   # below this → needs_review (in practice never reached)
```

The 0.92 entailment threshold is unusually high because DeBERTa-v3 is overconfident on this calibration band. The 0.75 contradiction threshold was lowered from 0.85 after an internal sweep showed it improved 3-class accuracy and contradiction recall without raising the false-support rate. The sweep ran on the gold set in `eval/benchmark/fixtures/gold/`, which is climate and energy claims rather than biomedical, so its numbers are not quoted here (see `CHALLENGES.md`).

**Self-verify trust guard.** If every retrieved evidence span for a claim came from the *same* document as the claim itself (the report), the verdict is in self-verify mode and `supported` confidences are capped at `self_verify_supported_cap` (0.80). The cap applies to `supported` only — `contradicted` and `insufficient_support` are not capped. See `TRUST_MODEL.md` for why.

Edge cases bypass the model entirely: zero evidence spans → `insufficient_support`; all-unresolvable anchors → `needs_review` (retrieval was skipped earlier, no broad fallback for this case).

### 5. Reviewer dashboard

`src/evidenceengine/api/routes/dashboard.py`, `templates/dashboard_*.html`

Server-rendered HTMX + Alpine.js + Jinja2. Shows all claims for a run sorted by severity (contradicted first, then needs_review, then insufficient_support, then supported). Click a claim to see verdict, confidence, NLI score breakdown, the verbatim source quote at its character offset, and per-document evidence linking via `verdict_evidence`. Approve, reject, or flag each verdict.

---

## Service-level layout

```
┌────────────────────────────────────────────────────────────┐
│                       HTTP Clients                          │
│     (browser / HTMX / curl / load_samples.py / tests)       │
└──────────────────────────┬─────────────────────────────────┘
                           │ REST + HTMX
┌──────────────────────────▼─────────────────────────────────┐
│                FastAPI Application Layer                    │
│   submit  /  packets  /  runs  /  dashboard  /  explain     │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│                  Pipeline Orchestrator                      │
│   (async background job, owns its own SQLAlchemy session)   │
└──┬────────────┬──────────────┬─────────────┬───────────────┘
   │            │              │             │
   ▼            ▼              ▼             ▼
┌──────┐   ┌────────┐    ┌──────────┐   ┌──────────────┐
│Parse │   │Extract │    │Retrieve  │   │Classify      │
│      │   │NLTK +  │    │BM25 +    │   │NLI cross-    │
│PyMuPDF│  │filters │    │BGE +     │   │encoder       │
│      │   │        │    │MiniLM    │   │(SciFact)     │
└──────┘   └────────┘    └──────────┘   └──────────────┘
                                              │
                                              ▼
                                     ┌──────────────────┐
                                     │   PostgreSQL 16   │
                                     │   (provenance)    │
                                     └──────────────────┘
```

The orchestrator owns its own `async_session_factory` session — it never inherits a request-scoped session. A separate `_mark_failed` function opens a fresh session to persist failures, isolating failure writes from a potentially corrupted primary session.

`RunVersion` status transitions: `pending` → `running` → `completed` / `failed`. The dashboard polls run status and renders per-stage progress bars with ETAs based on rolling per-stage wall-clock measurements.

---

## Performance characteristics (shared-cpu-2x, 4GB RAM)

| Stage | Typical | Notes |
|---|---|---|
| Parse | 2-5s | Skipped for tables on >50pp PDFs |
| Extract | <1s in thread pool | NLTK + regex |
| Retrieve | 5-30s | Dense skipped if corpus >3000 chunks |
| Classify | 15-60s | NLI on 25 claims * top-N spans, CPU |
| Total | 48-80s typical, 2-4 min on large PDFs | Bottleneck is NLI on shared CPU |

These numbers are from the developer environment. Runs on a busy shared machine took much longer (see Speed in `CHALLENGES.md`). A workstation with more cores cuts classify time roughly linearly with core count.

---

## Data model (provenance)

Key tables:

- `DocumentPacket` — one upload session (one report + N cited sources)
- `SourceDocument` — one PDF/DOCX file in a packet, with `is_report=True` for the report and `False` for cited papers
- `RunVersion` — one pipeline run on a packet, holds status + per-stage timing + pipeline config snapshot
- `Claim` — one extracted factual claim, links to its `SourceDocument` (the report) and `RunVersion`
- `CitationAnchor` — a citation marker inside a claim sentence; resolves to a specific `SourceDocument` or stays unresolved
- `EvidenceSpan` — a retrieved passage; carries `char_start`, `char_end`, `relevance_score`, `source_document_id`
- `Verdict` — one classification result for one claim; `verdict_type`, `confidence_score`, `reasoning`
- `VerdictEvidence` — join table linking a `Verdict` to the `EvidenceSpan`s that informed it
- `ReviewDecision` — human reviewer action (approve / reject / flag) on a verdict

The `VerdictEvidence → EvidenceSpan` chain is what lets the dashboard show "this verdict, justified by *this exact passage* from *this specific cited paper*." The recent dashboard source-doc filter (commit `8ecb201`) walks this chain to determine which claims belong to which uploaded document.

---

## Why no LLM in the hot path

Earlier revisions of EvidenceEngine used an LLM for both extraction and classification (the deleted `evidenceengine.classification.classifier` import is the fossil). Removing the LLM from the hot path bought:

- **Deterministic verdicts.** Same input → same verdict. No prompt drift, no provider-side model swap regressions.
- **No API key required to run the verifier.** The repo can be cloned and run without OpenRouter, OpenAI, or Gemini accounts.
- **Cost-free per-run inference.** The shared CPU server runs the full pipeline at no marginal cost.
- **Simpler trust model.** NLI confidence scores are the only knob. No "did the LLM hallucinate the verdict" failure mode.

What was lost: the LLM was better at handling ambiguous, qualitative claims ("the treatment was generally well-tolerated"). The NLI model returns `needs_review` more often on those. This is a deliberate trade-off in favor of conservatism. See `TRUST_MODEL.md`.

---

## Modes

The system has three retrieval modes that are selected automatically based on what the user uploaded and which citations resolved:

- **Mode A (self-verify):** No cited papers uploaded. The retriever pulls evidence from other paragraphs of the report itself. Useful for catching internal contradictions and unsupported assertions. `supported` verdicts are confidence-capped at 80%.
- **Mode B1 (resolved citation):** A claim's citation anchor resolves to a specific uploaded cited paper. Retrieval is scoped to that paper.
- **Mode B2 (unresolved citation, skip):** A claim has a citation anchor but it doesn't match any uploaded source. The claim is marked `needs_review` and skipped from broad retrieval to avoid spurious cross-document evidence.
- **Mode B3 (broad multi-doc):** No citation anchor in the claim sentence. The retriever searches across all uploaded cited papers.

Mode is recorded per-claim and surfaced in the dashboard.
