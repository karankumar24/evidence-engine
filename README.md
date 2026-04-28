# EvidenceEngine

Biomedical claim verification system. Upload a manuscript or systematic review plus the primary studies it cites — get back a verdict for each claim, anchored to specific evidence passages from your cited sources.

**Live demo:** https://evidenceengine.fly.dev

## What it does

1. Upload your manuscript or review (PDF or DOCX) plus the primary studies it cites.
2. The pipeline extracts factual claims (e.g. "Subcutaneous semaglutide reduced AF risk RR 0.77, 95% CI 0.60–0.99").
3. Citation markers (`(12)`, `[1]`, `Smith et al., 2023`) are resolved to your uploaded sources.
4. Evidence passages are retrieved from the cited source for each claim.
5. A local NLI model classifies each verdict — Supported / Contradicted / Insufficient Support / Needs Review — with provenance to the exact passage.

Optimized for biomedical and clinical research papers. Free open-access PDFs from PubMed Central, PLOS, Frontiers, BMC, and NEJM work best.

## Stack

- **Backend:** FastAPI + SQLAlchemy (async) + PostgreSQL
- **Extraction:** NLTK + heuristic filter (universal across scientific PDFs)
- **Retrieval:** BM25 + dense (`BAAI/bge-base-en-v1.5`) + cross-encoder rerank (`cross-encoder/ms-marco-MiniLM-L6-v2`)
- **NLI:** SciFact-fine-tuned `cross-encoder/nli-deberta-v3-small` — local inference, zero LLM calls in the hot path
- **LLM (optional):** Gemini / Groq fallback chain for the on-demand explanation endpoint only
- **Frontend:** HTMX + Alpine.js + Tailwind v4 (server-side rendered)
- **Deploy:** Fly.io (shared-cpu-2x, persistent volume for the fine-tuned NLI model)

## Local dev

```bash
# Requires: Python 3.13, PostgreSQL, Node 20
cp .env.example .env
uv sync
npm ci && npm run build:css
alembic upgrade head
uvicorn evidenceengine.api.app:app --reload
```

## CI

GitHub Actions runs `ruff`, `mypy`, and `pytest` on every push. See `.github/workflows/test.yml`.

## Out of scope

- ML / CS papers — NLI is fine-tuned for biomedical claims; accuracy degrades 15–25 percentage points on ML/CS text.
- Long policy or financial documents — extraction works but the 25-claim cap loses 99% of claims on documents over ~50 pages.
