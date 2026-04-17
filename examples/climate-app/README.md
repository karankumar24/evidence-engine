# EvidenceEngine — Climate Reference App

A reference application demonstrating the EvidenceEngine pipeline on climate and environmental
reports. Upload a report and its cited sources; the pipeline extracts claims, retrieves
supporting evidence, and produces claim-by-claim verdicts (SUPPORTED / CONTRADICTED /
INSUFFICIENT_SUPPORT / NEEDS_REVIEW) with full provenance.

This directory ships with two sample report packets so you can see end-to-end results
without writing any content or curl commands.

---

## What You Will See

Running this quickstart exercises the full pipeline:

1. **Upload** — report + source documents uploaded as a packet via the REST API
2. **Ingest** — files parsed into text blocks with character-level offsets
3. **Extract** — LLM extracts individual factual claims from the report
4. **Retrieve** — BM25 retrieval finds the most relevant evidence spans from source docs
5. **Classify** — LLM classifies each claim against retrieved evidence
6. **Dashboard** — verdict table viewable at `http://localhost:8000/dashboard`

---

## Prerequisites

- **Python 3.12+**
- **uv** (fast Python package manager) — install with: `pip install uv`
- **Docker + Docker Compose** — for the PostgreSQL database
- **OpenAI API key** — for claim extraction and classification

---

## Quickstart (clone to verdicts in 6 steps)

### Step 1 — Clone and install

```bash
git clone https://github.com/your-org/vibebuild.git
cd vibebuild
uv sync
```

This installs all Python dependencies (including `httpx`, `fastapi`, `sqlalchemy`,
`alembic`, and all pipeline packages) into a local virtual environment.

### Step 2 — Start PostgreSQL

```bash
docker-compose up -d
```

This starts a PostgreSQL 16 container on port 5432. The `docker-compose.yml` in the project
root defines the service with the credentials used in `src/evidenceengine/core/config.py`.

Verify the database is running:

```bash
docker-compose ps
```

Expected: `State` column shows `Up` for the postgres service.

### Step 3 — Run database migrations

```bash
uv run alembic upgrade head
```

This applies all Alembic migrations and creates the required tables (`document_packets`,
`source_documents`, `text_blocks`, `claims`, `run_versions`, `verdicts`,
`evidence_spans`, `review_decisions`).

Expected output ends with: `INFO  [alembic.runtime.migration] Running upgrade ... -> ...`

### Step 4 — Set your OpenAI API key

```bash
export OPENAI_API_KEY=sk-...
```

Or create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

The server reads `.env` automatically via `python-dotenv` at startup. Without this key,
the extraction and classification stages will fail.

### Step 5 — Start the API server

```bash
uv run uvicorn evidenceengine.api.app:app --reload
```

Expected: `INFO: Application startup complete.` and `INFO: Uvicorn running on http://127.0.0.1:8000`

The `--reload` flag restarts the server automatically when source files change (development mode).
For production use omit `--reload` and run with multiple workers.

### Step 6 — Load sample packets and run the pipeline

Open a second terminal (the server must remain running) and execute:

```bash
python examples/climate-app/load_samples.py
```

Expected output:

```
[carbon_budget_brief]
  Uploading: Carbon Budget Policy Brief
  packet_id:      <uuid>
  run_version_id: <uuid>
  Waiting for pipeline... COMPLETED

[ipcc_ar6_excerpt]
  Uploading: IPCC AR6 Excerpt
  packet_id:      <uuid>
  run_version_id: <uuid>
  Waiting for pipeline... COMPLETED

Done. Both packets completed successfully.
Open http://localhost:8000/dashboard to view results in the browser.
```

Pipeline runs take 30–90 seconds per packet depending on OpenAI API latency.

### Step 7 — View results in the dashboard

Open your browser to:

```
http://localhost:8000/dashboard
```

You will see a review queue listing claims from both packets. Click any claim to see:
- The extracted claim text with its citation
- The verdict (SUPPORTED / CONTRADICTED / INSUFFICIENT_SUPPORT / NEEDS_REVIEW)
- Confidence score and LLM reasoning
- The exact evidence spans retrieved from source documents

---

## Sample Packets Included

| Packet | Domain | Claims | Expected Verdict Mix |
|--------|--------|--------|----------------------|
| `ipcc_ar6_excerpt` | Physical climate science (IPCC AR6) | 6–8 | SUPPORTED, CONTRADICTED, INSUFFICIENT_SUPPORT |
| `carbon_budget_brief` | Carbon budget / mitigation policy | 6 | SUPPORTED, INSUFFICIENT_SUPPORT |

Both packets use synthetic content modelled on publicly available 2022–2023 climate data.
The IPCC AR6 excerpt packet intentionally includes one contradicted claim (methane ppb value)
to demonstrate the pipeline's ability to catch numerical discrepancies.

All content is stored as plain `.txt` files — no binary formats, no external downloads.

---

## Using load_samples.py

```
usage: load_samples.py [-h] [--base-url BASE_URL]

options:
  -h, --help           show this help message and exit
  --base-url BASE_URL  Base URL of the running EvidenceEngine API
                       (default: http://localhost:8000)
```

Point at a remote or staging instance:

```bash
python examples/climate-app/load_samples.py --base-url https://staging.example.com
```

The script exits with code `0` if both packets complete successfully, `1` if either fails.
This makes it suitable as a smoke test in CI pipelines.

---

## Troubleshooting

### "Connection refused" or "RequestError"

The API server is not running. Make sure you have completed Step 5:

```bash
uv run uvicorn evidenceengine.api.app:app --reload
```

### "Could not connect to server" (database errors at startup)

PostgreSQL is not running. Check with `docker-compose ps` and restart with:

```bash
docker-compose up -d
```

### Pipeline status is "failed"

Check the server terminal for error output. Common causes:

- **Missing OPENAI_API_KEY** — set the environment variable and restart the server
- **OpenAI quota exceeded** — check your OpenAI usage dashboard
- **Database migration not applied** — run `uv run alembic upgrade head`

### Port 5432 already in use

Another PostgreSQL instance is occupying the port. Stop it or change the port in
`docker-compose.yml` and update the `DATABASE_URL` in your `.env`.

### Port 8000 already in use

```bash
uv run uvicorn evidenceengine.api.app:app --reload --port 8001
python examples/climate-app/load_samples.py --base-url http://localhost:8001
```

### Checking server logs

The server prints pipeline stage progress to stdout. Look for lines like:

```
INFO: Orchestrator: starting pipeline for run <uuid>
INFO: Orchestrator: extraction complete — N claims found
INFO: Orchestrator: classification complete
INFO: Orchestrator: pipeline completed
```

---

## Running Your Own Report

To verify a custom report against your own sources:

```bash
# Upload your own packet
curl -X POST http://localhost:8000/api/packets/ \
  -F "report=@/path/to/your/report.pdf" \
  -F "sources=@/path/to/source1.pdf" \
  -F "sources=@/path/to/source2.pdf"

# The response includes a packet_id — use it to trigger the pipeline
curl -X POST http://localhost:8000/api/packets/<packet_id>/run \
  -H "Content-Type: application/json" \
  -d '{}'

# Poll for status
curl http://localhost:8000/api/runs/<run_version_id>
```

Then visit the dashboard to review results.
