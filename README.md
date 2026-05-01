# EvidenceEngine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Live demo](https://img.shields.io/badge/demo-evidenceengine.fly.dev-brightgreen)](https://evidenceengine.fly.dev)

**A tool that fact checks biomedical research papers against the studies they cite.**

When researchers publish a review article or systematic review, they cite dozens of studies. Sometimes a citation is a stretch. Sometimes the cited study actually says the opposite. Catching this manually means reading every cited paper, and most reviewers don't have time. EvidenceEngine reads the papers for you and flags claims that don't hold up.

![Dashboard claim detail](docs/screenshots/dashboard-claim.png)

You upload your paper plus the studies it cites. The tool pulls out each factual claim, finds the relevant passages in the cited papers, and gives a verdict for each claim with a direct quote of the supporting (or contradicting) text. Verdicts come in four kinds: **Supported**, **Contradicted**, **Insufficient Support**, or **Needs Review**.

**Try it live:** https://evidenceengine.fly.dev

---

## How it works

```
PDF upload  →  Claims pulled out  →  Evidence retrieved  →  Verdict assigned  →  Reviewer dashboard
```

1. **Read the PDF.** Open source PDF parser pulls out the text in reading order, keeping page and paragraph metadata.
2. **Pull out the claims.** A sentence tokenizer plus a stack of filters (skip bibliographies, methods boilerplate, glossary entries, page headers, license text) leaves you with the actual factual statements. Caps at 25 claims per run, biased toward the ones with citations.
3. **Find the evidence.** Two search methods run together. Keyword search (BM25) finds passages with literal word overlap. Vector search (a small embedding model) finds passages that mean the same thing in different words. A reranker scores the combined candidates and picks the best ones.
4. **Decide the verdict.** A small classification model that was trained on a public biomedical fact checking dataset (SciFact) reads each claim alongside the retrieved evidence and outputs one of the four verdicts with a confidence score.
5. **Show it to the reviewer.** A clean dashboard lets you go through claim by claim, see the exact source quotes, and approve or reject each verdict.

The whole pipeline runs locally. No external API calls happen during classification. The only optional outside call is when you click "Explain this verdict" on a single claim, which sends just that claim and its evidence to a language model for a one paragraph plain English explanation.

---

## What it can and can't do

**Can:**
- Fact check biomedical papers against their cited primary studies.
- Catch citations that don't actually support the claim.
- Provide direct quotes from sources so you can verify the verdict yourself.
- Run on a small server with no API keys (the explanation feature is optional).

**Can't:**
- Fact check papers in fields outside biomedicine (machine learning, finance, policy). The classification model was trained on biomedical claims and accuracy drops 15 to 25 percentage points elsewhere.
- Handle very long documents well. The 25 claim cap means a 50 page review only gets a sample of claims checked.
- Verify claims against papers you didn't upload. If your paper cites a study but you didn't upload that study's PDF, the system marks the claim as needing review rather than guessing.

---

## How well it works

Tested on a 220 case internal benchmark covering all four verdict types.

| Metric | Score | What it means |
|---|---|---|
| 3-class accuracy | 84.8% | How often the verdict was right when the answer was supported, contradicted, or insufficient |
| False support rate | 9.1% | How often a weak or wrong claim got marked supported anyway. Lower is better. |
| Contradiction recall | 72.7% | Of the claims that were actually contradicted by their sources, how many the system caught |

Speed varies a lot because the deploy server is shared. A typical run takes 3 to 15 minutes for one paper. Re-uploading the same PDF is much faster because the embeddings are cached.

---

## Built with

| Layer | Tool |
|---|---|
| Web framework | FastAPI with async SQLAlchemy and PostgreSQL |
| Frontend | HTMX, Alpine.js, Tailwind CSS (no single page app) |
| PDF reading | PyMuPDF |
| Keyword search | BM25 (bm25s library) |
| Vector search | BAAI/bge-base-en-v1.5 |
| Reranker | cross-encoder/ms-marco-MiniLM-L6-v2 |
| Verdict classifier | cross-encoder/nli-deberta-v3-small fine tuned on SciFact |
| Optional explainer | Gemini (only used for the on demand explanation button) |
| Hosting | Fly.io shared CPU, 4 GB RAM, with a persistent volume for the model |

---

## Run it locally

You need Python 3.13, Node 20, and a local PostgreSQL.

```bash
uv sync
npm ci && npm run build:css
alembic upgrade head
uvicorn evidenceengine.api.app:app --reload
```

Open http://localhost:8000 and upload a PDF.

Full setup with environment variables and production deploy steps: [`docs/DEPLOY.md`](docs/DEPLOY.md).

For the architecture and design decisions: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). For why weak evidence always gets marked insufficient instead of supported: [`docs/TRUST_MODEL.md`](docs/TRUST_MODEL.md). For the full list of models and what each one does: [`docs/MODELS_AND_TERMS.md`](docs/MODELS_AND_TERMS.md).

## License

MIT. See [LICENSE](LICENSE).
