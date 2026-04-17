# EvidenceEngine Architecture

## Overview

EvidenceEngine is an automated fact-checking pipeline that ingests a document
(the *report*) and its cited source files, then produces a claim-by-claim audit
trail: every factual assertion in the report is extracted, matched against the
evidence it cites, and classified as `supported`, `contradicted`,
`insufficient_support`, or `needs_review`. The result is stored with full
provenance so a human reviewer can trace any verdict back to the exact character
offset in the source document that informed it.

The system is designed for document reviewers, policy researchers, and
fact-checkers who need to process reports faster than manual cross-checking
allows, but who still want to apply human judgment before acting on a verdict.
EvidenceEngine is deliberately conservative: a claim is `supported` only when
evidence directly and unambiguously confirms it; anything else routes to
`insufficient_support` or human review.

What distinguishes EvidenceEngine from generic RAG pipelines is the provenance
data model. Every evidence span carries the character offsets of the text it
came from. Every verdict is linked to the exact spans that informed it via a
`VerdictEvidence` join table. A reviewer can click a verdict and see the raw
sentence from the source document — not a paraphrase, the verbatim text at the
verified offset.

---

## System Components

### Component Map

```
┌─────────────────────────────────────────────────────────────┐
│                        HTTP Clients                          │
│         (curl / load_samples.py / browser / tests)           │
└───────────────────────────┬─────────────────────────────────┘
                             │ REST API  /  HTMX
┌───────────────────────────▼─────────────────────────────────┐
│                 FastAPI Application Layer                     │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  │
│  │  Packets │  │  Pipeline │  │   Runs   │  │ Dashboard │  │
│  │  Routes  │  │  Routes   │  │  Routes  │  │  Routes   │  │
│  └────┬─────┘  └─────┬─────┘  └────┬─────┘  └─────┬─────┘  │
└───────┼──────────────┼─────────────┼───────────────┼─────────┘
        │              │             │               │
┌───────▼──────────────▼─────────────▼───────────────▼─────────┐
│                       Pipeline Services                        │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌────────────┐   │
│  │ Ingestion│  │Extraction │  │Retrieval │  │Classifica- │   │
│  │  Service │  │  Service  │  │ Service  │  │tion Service│   │
│  └────┬─────┘  └─────┬─────┘  └────┬─────┘  └─────┬──────┘   │
└───────┼──────────────┼─────────────┼───────────────┼───────────┘
        │              │             │               │
┌───────▼──────────────▼─────────────▼───────────────▼───────────┐
│                    Pipeline Orchestrator                         │
│         (async background job — RunVersion lifecycle)            │
└───────────────────────────────────┬─────────────────────────────┘
                                    │
        ┌───────────────────────────┼──────────────────────┐
        │                           │                      │
┌───────▼──────────┐  ┌─────────────▼──────────┐  ┌───────▼──────────┐
│   PostgreSQL 16   │  │  OpenAI GPT-4o-mini    │  │  BM25s + Cross-  │
│   (persistence)   │  │  (extraction +         │  │  Encoder Reranker│
│                   │  │   classification LLM)  │  │  (retrieval)     │
└───────────────────┘  └────────────────────────┘  └──────────────────┘
```

### Component Descriptions

**FastAPI Application Layer** — The HTTP interface. Four route groups cover
the full lifecycle: `packets` routes handle file upload and packet inspection;
`pipeline` routes expose individual pipeline stages so callers can trigger
each step independently; `runs` routes trigger the full four-stage orchestrated
pipeline and let callers poll progress; `dashboard` routes serve the HTMX
reviewer interface. Auto-generated OpenAPI docs are available at `/docs`.

**Ingestion Pipeline** — Accepts PDF or DOCX files, parses each to a list of
text blocks where every block carries `char_start` and `char_end` offsets
into the document's raw text. The parse-then-accept pattern means all files
are fully parsed before any database row is written — a failed parse on one
file rolls back the entire upload and removes partially-written uploads.

**Extraction Pipeline** — Uses an OpenAI GPT-4o-mini LLM call to decompose
the report's text blocks into atomic, verifiable claims. A regex-based
citation detector first marks all `[1]`, `[2]` markers; the extractor resolves
each claim's citation reference to a `SourceDocument` row. Any citation that
cannot be resolved to a known source document is written as a `CitationAnchor`
with `status='unresolvable_anchor'` — it is never silently dropped.

**Retrieval Pipeline** — For each resolved `CitationAnchor`, retrieves
evidence from the cited `SourceDocument` only (not all documents). Builds a
disk-backed BM25 index per source document, queries it with the claim text,
then re-ranks the top candidates with a `sentence-transformers` cross-encoder
model. The top-N scoring spans become `EvidenceSpan` rows, each carrying
`char_start`, `char_end`, and `relevance_score`.

**Classification Pipeline** — Classifies each claim against its retrieved
evidence spans. The LLM receives the claim text, all evidence spans, and an
explicit prompt that instructs it to return `insufficient_support` for partial
or topically-related-but-not-confirming evidence. A confidence threshold
function (`apply_confidence_threshold`) then checks the returned
`confidence_score`: if it falls below the configured threshold the verdict is
overridden to `needs_review`. Two edge cases bypass the LLM entirely: zero
evidence spans → `insufficient_support`; unresolvable citation anchor →
`needs_review`.

**Pipeline Orchestrator** — Runs all four stages as a single async background
job triggered by `POST /api/packets/{id}/runs`. The orchestrator owns its own
SQLAlchemy async session (`async_session_factory`) — it never inherits a
request-scoped session — and writes `RunVersion` status (`pending` →
`running` → `completed` / `failed`) and model version metadata throughout
execution. A separate `_mark_failed` function opens a fresh session to persist
failures, isolating failure writes from a potentially corrupted primary session.

**Reviewer Dashboard** — A server-rendered HTMX + Alpine.js + Jinja2
interface for human reviewers. Shows all claims for a run, sorted by severity
(contradicted first, then needs_review, then insufficient_support, then
supported). Reviewers can accept, override, or flag each verdict; each action
writes a `ReviewDecision` row without modifying the original `Verdict` row.
HTMX OOB swap updates the queue row status span in-place to preserve HTMX
bindings.

**Evaluation Harness** — A standalone CLI (`python -m eval.runner`) that runs
the four core metrics — exact-match accuracy, false-positive rate,
false-negative rate, and confidence calibration — against a fixture dataset of
benchmark cases. Operates without a running web server or database, making it
suitable for CI quality gates.

---

## Data Flow

### End-to-End Pipeline Walk-Through

#### Step 1 — Document Upload (Ingestion)

**Trigger:** `POST /api/packets` with multipart form data containing a report
file and one or more source files.

**Input:** A PDF or DOCX report and one or more PDF/DOCX source documents.

**Processing:** All files are parsed to text blocks *before* any database write
(parse-then-accept atomicity). Each text block carries `char_start` and
`char_end` offsets into the document's `raw_text`. The offset integrity
invariant is enforced at parse time: `raw_text[char_start:char_end] == block.text`
must hold for every block, or the parse raises a `ValueError`. If any file
fails to parse, the entire upload is rolled back and no rows are committed.

**Output:** A `DocumentPacket` row and multiple `SourceDocument` rows in
PostgreSQL. The report `SourceDocument` has `is_report=True`; cited sources
have `is_report=False`. Each `SourceDocument` stores `parsed_content` as a
JSONB array of text blocks.

**Key invariant:** offset integrity is checked at parse time, not via a DB
constraint — the invariant is language-level not schema-level.

---

#### Step 2 — Claim Extraction

**Trigger:** `POST /api/packets/{id}/extract` (or automatically when the
orchestrator runs).

**Input:** The report `SourceDocument.parsed_content` blocks.

**Processing:**
1. Regex citation detector finds all `[N]` markers in report text.
2. LLM (GPT-4o-mini) decomposes the report into atomic, independently
   verifiable claims — one claim per sentence-level factual assertion.
3. Each claim's citation reference is resolved to a `SourceDocument` row by
   matching citation number to upload order.
4. Character offsets for each claim's location in the report text are
   recovered from the parsed blocks.

**Output:** `Claim` rows (with `char_start`/`char_end` offsets in report text,
`status='pending'`) and `CitationAnchor` rows linking each claim to its cited
`SourceDocument`. Unresolvable citations write a `CitationAnchor` with
`status='unresolvable_anchor'`.

**Key invariant:** Every citation reference in the report is written as a
`CitationAnchor` row, resolved or not — no anchor is silently dropped.

---

#### Step 3 — Evidence Retrieval

**Trigger:** `POST /api/packets/{id}/retrieve` (or automatically via orchestrator).

**Input:** `Claim` rows with resolved `CitationAnchor` rows pointing to
specific `SourceDocument` rows.

**Processing:**
1. For each resolved `CitationAnchor`, build (or load from disk) a BM25 index
   of the cited `SourceDocument`'s text blocks.
2. Query the BM25 index with the claim text to retrieve top-K candidates.
3. A `sentence-transformers` cross-encoder reranker re-scores the candidates
   against the claim text.
4. Top-N spans by reranker score are selected as evidence.

**Output:** `EvidenceSpan` rows, each recording `char_start`, `char_end`,
`raw_text`, `relevance_score`, and `rank` within the cited source document.
Recall@K metrics are logged to `RunVersion.pipeline_config`.

**Key decision:** Retrieval searches only the *cited* `SourceDocument` (not all
uploaded sources). This preserves citation semantics — evidence comes from the
source the author cited, not from any conveniently matching passage elsewhere.

---

#### Step 4 — Verdict Classification

**Trigger:** `POST /api/packets/{id}/classify` (or automatically via orchestrator).

**Input:** `Claim` rows and their associated `EvidenceSpan` rows.

**Processing:**
1. **Edge case — zero evidence spans:** If no `EvidenceSpan` rows exist for a
   claim, assign `insufficient_support` with `model_name='none'` — no LLM call.
2. **Edge case — unresolvable anchor:** If the claim's `CitationAnchor` is
   unresolvable, assign `needs_review` with `model_name='none'` — no LLM call.
3. **Standard case:** LLM classifies the claim against its evidence spans using
   the structured-output API (`beta.chat.completions.parse`). The prompt
   includes explicit disambiguation instructions for partial evidence.
4. `apply_confidence_threshold(verdict_type, confidence_score, threshold)` is
   applied. If `confidence_score < threshold`, `verdict_type` is overridden to
   `needs_review` regardless of the LLM's original label.

**Output:** `Verdict` rows (`verdict_type`, `confidence_score`, `reasoning`) and
`VerdictEvidence` join rows linking each `Verdict` to the `EvidenceSpan` rows
that informed it.

**Key decision:** Field order in `VerdictClassificationResponse` is
`reasoning → verdict_type → confidence_score`. This forces the LLM to produce
chain-of-thought reasoning before assigning a label, reducing shortcut label
assignment.

---

#### Step 5 — Orchestration

**Trigger:** `POST /api/packets/{id}/runs`

**Processing:** An async background task runs steps 2–4 sequentially. The
orchestrator:
- Creates a `RunVersion` row with `status='pending'`
- Updates status to `'running'` before stage execution starts
- Records `model_versions` dict (LLM model name, cross-encoder model name) in
  the `RunVersion` after each stage
- On success: updates `status='completed'`, sets `completed_at`, writes
  `pipeline_config` JSONB with per-stage metrics
- On failure: `_mark_failed` opens a *separate* fresh session to write
  `status='failed'` and the error message — this session is isolated from any
  corrupted primary session

**Output:** `RunVersion` row with `status='completed'` (or `'failed'`),
populated `model_versions`, and `pipeline_config` containing metrics like
`failed_claim_count` and recall@K values.

---

## Provenance Data Model

### Entity Relationship Overview

```
DocumentPacket
  ├── SourceDocument  (is_report=True  — the report being verified)
  │     └── [parsed_content JSONB — text blocks with char offsets]
  └── SourceDocument  (is_report=False — one per cited source document)
        └── EvidenceSpan  (retrieved text span for a specific claim)
              ├── char_start, char_end  (offsets into SourceDocument.raw_text)
              └── relevance_score, rank

RunVersion  (tracks a single pipeline execution on a DocumentPacket)
  └── Claim  (one atomic factual statement extracted from the report)
        ├── char_start, char_end  (offsets into report SourceDocument.raw_text)
        ├── CitationAnchor  → SourceDocument  (resolves citation [N] to a file)
        │     └── status: 'resolved' | 'unresolvable_anchor'
        ├── Verdict  (supported | contradicted | insufficient_support | needs_review)
        │     ├── verdict_type, confidence_score, reasoning
        │     └── VerdictEvidence  → EvidenceSpan  (which spans informed this verdict)
        └── ReviewDecision  (human reviewer action: accept / override / flag)
              └── Preserves original Verdict — never mutates it
```

### Key Design Decisions

**`is_report` boolean on `SourceDocument`** — A single `SourceDocument` model
with an `is_report` flag rather than separate `Report` and `Source` tables.
Simpler to query (one join, not two), avoids ambiguous FK chains, and makes the
"report is also a document" semantic explicit.

**Offset integrity invariant** — `raw_text[char_start:char_end] == block.text`
is enforced at parse time with a `ValueError`, not as a database constraint.
This means malformed content is rejected before any I/O, making the invariant
impossible to violate silently.

**`CitationAnchor` always written** — Whether a citation resolves or not, a
`CitationAnchor` row is always written. This satisfies the CLAIM-06 requirement:
unresolvable anchors are tracked explicitly so reviewers know a claim had a
citation that couldn't be matched, rather than the claim appearing citation-free.

**`VerdictEvidence` join table** — Linking `Verdict` to `EvidenceSpan` through
a join table (rather than a FK on `Verdict`) enables the "which evidence spans
produced this verdict" query without denormalizing evidence text into the
`Verdict` row, and allows a single span to inform multiple verdicts if needed.

---

## API Surface

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/packets` | Upload report + source files; returns `packet_id` |
| `GET` | `/api/packets/{id}` | Retrieve packet metadata and source documents |
| `POST` | `/api/packets/{id}/extract` | Trigger claim extraction (Step 2) |
| `POST` | `/api/packets/{id}/retrieve` | Trigger evidence retrieval (Step 3) |
| `POST` | `/api/packets/{id}/classify` | Trigger verdict classification (Step 4) |
| `POST` | `/api/packets/{id}/runs` | Trigger full four-stage pipeline (Steps 2–4) |
| `GET` | `/api/runs/{id}` | Poll pipeline run status and progress |
| `GET` | `/dashboard` | Reviewer dashboard (HTMX server-rendered) |
| `POST` | `/dashboard/verdicts/{verdict_id}/review` | Submit a reviewer decision |
| `GET` | `/dashboard/queue` | Partial queue refresh (HTMX target) |

Individual stage endpoints (`/extract`, `/retrieve`, `/classify`) exist for
development and debugging. Production workflows should use `/runs`, which
handles the full lifecycle including `RunVersion` tracking.

---

## Technology Stack

| Layer | Technology | Why |
|-------|------------|-----|
| Web framework | FastAPI + asyncpg | Async-native; type-safe request/response; auto OpenAPI docs |
| ORM | SQLAlchemy (async) | Complex FK chains in provenance model; async session management |
| Database | PostgreSQL 16 | JSONB for `parsed_content`; reliable FK enforcement; Docker-composable |
| BM25 retrieval | bm25s | Disk-backed index; no vector DB required in v1; fast keyword matching |
| Reranking | sentence-transformers cross-encoder | Re-scores BM25 candidates with semantic similarity; improves recall@K |
| LLM | OpenAI GPT-4o-mini | Structured output via `beta.chat.completions.parse`; cost-efficient |
| Dashboard | HTMX + Alpine.js + Jinja2 | Progressive enhancement; no JS build step; CDN-served Tailwind |
| Package manager | uv | Fast, lockfile-based; compatible with setuptools; reproducible installs |
| Evaluation | Pure Python CLI (`eval.runner`) | No web server dependency; CI-runnable quality gate |

---

## Related Documents

- See [docs/TRUST_MODEL.md](TRUST_MODEL.md) for the verdict taxonomy and
  confidence routing design — including the weak-evidence rule and the
  `apply_confidence_threshold` mechanism.
- See [eval/README.md](../eval/README.md) for the evaluation harness — how
  verdict quality is measured and regression-tested.

---

*For questions a reader of this document should be able to answer:*

- **"Where does the claim text come from?"** — The LLM extracts it from the
  report's `parsed_content` blocks during the Extraction stage (Step 2).
- **"How does the system know which source document to search?"** — The
  `CitationAnchor` resolves the claim's `[N]` citation reference to a
  specific `SourceDocument` row; retrieval searches only that document.
- **"What happens when a citation can't be resolved?"** — A `CitationAnchor`
  with `status='unresolvable_anchor'` is written, and the claim is assigned
  `needs_review` during classification (no LLM call needed).
- **"How is evidence linked back to verdicts?"** — Via the `VerdictEvidence`
  join table: each `Verdict` row is linked to the `EvidenceSpan` rows that
  the LLM saw when producing that verdict.
