# EvidenceEngine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Fact-checks biomedical research papers against the studies they cite.**

Upload a draft + the cited PDFs. The tool extracts every factual claim, finds the relevant passages in your cited papers, and verdicts each one as **Supported**, **Contradicted**, **Insufficient Support**, or **Needs Review** with a direct quote of the supporting text.

No LLM in the verdict path. Runs locally. No API key required.

---

## Why

Citation distortion in biomedical reviews runs 11 to 15 percent in published studies of the issue. Manually auditing 80 citations in a 30-page review takes a day. This tool routes attention to the suspicious ones in minutes. A human still confirms every verdict; the tool is an attention router, not a judge.

**Useful for:** evidence-synthesis postdocs, peer reviewers, grad students self-auditing a draft.
**Not useful for:** non-biomedical papers, anything fully automated, anyone without the cited PDFs.

---

## Works best with

- Open-access biomedical papers from PMC / PLOS / Frontiers / BMC / NEJM
- Vancouver-style citations (`[1]`, `[1-3]`)
- Born-digital text PDFs, 5 to 30 pages
- Uploaded as draft + 3 to 5 cited primary studies

**Will fail or mislead on:** scanned PDFs (no OCR), math-heavy papers, multi-column tables, non-biomedical domains (15 to 25pp accuracy drop).

---

## Pipeline

```
PDF → claims (NLTK + filters) → BM25 + BGE retrieval + reranker → SciFact-tuned NLI verdict → reviewer dashboard
```

| Stage | Tool |
|---|---|
| PDF parsing | PyMuPDF |
| Claim extraction | NLTK + heuristic filters |
| Retrieval | BM25 + BAAI/bge-base-en-v1.5 |
| Reranker | cross-encoder/ms-marco-MiniLM-L6-v2 |
| Verdict | cross-encoder/nli-deberta-v3-small fine-tuned on SciFact |
| Web | FastAPI, HTMX, Alpine, Tailwind, PostgreSQL |

---

## Run locally

```bash
uv sync
npm ci && npm run build:css
alembic upgrade head
uvicorn evidenceengine.api.app:app --reload
```

Open http://localhost:8000 and upload a PDF. Full setup: [`docs/DEPLOY.md`](docs/DEPLOY.md).

---

## Deeper docs

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — pipeline stages, code paths
- [`docs/TRUST_MODEL.md`](docs/TRUST_MODEL.md) — NLI thresholds + self-verify cap
- [`docs/MODELS_AND_TERMS.md`](docs/MODELS_AND_TERMS.md) — every model in plain English
- [`CHALLENGES.md`](CHALLENGES.md) — honest known limitations
- [`ATTRIBUTIONS.md`](ATTRIBUTIONS.md) — model + dataset licenses

---

## License

Project code: **MIT** (see [LICENSE](LICENSE)).

Bundled SciFact (`eval/data/scifact/`) is **CC BY-NC 2.0**. PyMuPDF is **AGPL 3.0**. See [`ATTRIBUTIONS.md`](ATTRIBUTIONS.md) before redistributing.
