# EvidenceEngine

Provenance-aware document verification system. Upload a PDF report, submit a claim, and get back a verdict with evidence anchored to specific passages.

**Live demo:** https://evidenceengine.fly.dev

## What it does

1. Upload a PDF (annual report, research paper, filing)
2. Submit a claim to verify ("Revenue grew 12% YoY")
3. Get a verdict — Supported / Refuted / Insufficient Evidence — with exact paragraph citations

## Stack

- **Backend:** FastAPI + SQLAlchemy (async) + PostgreSQL
- **LLM:** OpenRouter (multi-model fallback chain)
- **Retrieval:** BM25 + cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L6-v2`)
- **Frontend:** HTMX + Alpine.js + Tailwind v4 (server-side rendered)
- **Deploy:** Fly.io (scale-to-zero, persistent volume for uploads/indexes)

## Local dev

```bash
# Requires: Python 3.13, PostgreSQL, Node 20
cp .env.example .env   # fill in LLM_API_KEY
uv sync
npm ci && npm run build:css
alembic upgrade head
uvicorn evidenceengine.api.app:app --reload
```

## CI

GitHub Actions runs `ruff`, `mypy`, and `pytest` on every push. See `.github/workflows/test.yml`.
