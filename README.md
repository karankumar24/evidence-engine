# EvidenceEngine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**A tool that fact checks biomedical research papers against the studies they cite.**

When researchers publish a review article or systematic review, they cite dozens of studies. Sometimes a citation is a stretch. Sometimes the cited study actually says the opposite. Catching this manually means reading every cited paper, and most reviewers don't have time. EvidenceEngine reads the papers for you and flags claims that don't hold up.

You upload your paper plus the studies it cites. The tool pulls out each factual claim, finds the relevant passages in the cited papers, and gives a verdict for each claim with a direct quote of the supporting (or contradicting) text. Verdicts come in four kinds: **Supported**, **Contradicted**, **Insufficient Support**, or **Needs Review**.

The project is a portfolio piece. Run it locally to try it (instructions below). There is no public demo right now.

---

## The real problem this solves (brutal honesty)

Systematic reviewers, evidence-synthesis researchers, and meta-analysts spend **most of their time reading cited papers to confirm that each citation actually supports the claim it sits next to**. A senior reviewer auditing a 30-page review with 80 citations is doing 80 manual cross-checks. Each check is "open the cited PDF, search for the relevant passage, decide if the original paper said what the review says it said." Most reviewers do not have time, so they spot-check. Citations that misstate the source slip through.

This is a real and documented problem. Citation distortion in biomedical reviews has been measured at 11 to 15 percent of cited claims in published studies of the issue. That number does not include the much larger pile of citations that are technically true but cherry-picked or stretched.

EvidenceEngine attacks the *audit step* of that workflow. Upload your draft and the cited primary studies, and the tool surfaces every claim that does not hold up against its own cited evidence. A human still reviews every verdict; the tool just routes attention to the suspicious ones.

**Who this is actually useful for, today, with no edits:**
- An evidence-synthesis postdoc finishing a systematic review who already has all cited PDFs in Zotero. Drag them into the upload box, get a list of "this citation is fine, this one is a stretch, this one contradicts your claim."
- A peer reviewer doing a thorough pass on a manuscript and the cited primary literature. Faster triage than a manual citation audit.
- A grad student who wants a second pass on their own draft before submission, looking for places where they over-claimed against a cited study.

**Who this is NOT useful for:**
- Anyone working outside biomedicine. The classifier was trained on biomedical claims; non-bio domains drop 15 to 25 percentage points in accuracy.
- People who want a fully automated "is this paper good" verdict. The tool is an attention router, not a judge. Every verdict needs a human to confirm.
- People who don't have the cited PDFs. The tool can only verify against papers you upload. It does not crawl the web for missing sources.

---

## What kinds of PDFs work best

**Best results:**
- Open-access biomedical papers from PubMed Central (PMC), PLOS, Frontiers, BMC, NEJM open-access. These are well-formatted text PDFs born digital.
- Reviews and systematic reviews with clear Vancouver-style numeric citations (`[1]`, `[1-3]`).
- Cited primary studies that are themselves text PDFs with extractable abstracts and results sections.
- Papers in the 5 to 30 page range. Long reviews still work but the 25-claim cap means only a sample of claims gets verified per run.

**Works but worse:**
- Papers with APA-style inline citations like `(Smith et al., 2023)` if the reference list is parseable. The anchor resolver does fuzzy matching against the references, which works when first author and year are extractable.
- Older biomedical papers with non-standard PDF formatting. Extraction is noisier but usually still works.
- Climate science and adjacent biomedical-friendly domains. Calibration is mid; verdicts are usually directionally right.

**Will be wrong or refuse to work:**
- Scanned PDFs with no text layer. There is no OCR fallback shipped. The system extracts zero claims and the run is empty.
- Math-heavy papers (theoretical physics, ML papers with heavy LaTeX). Equations get rendered into glyph soup that the extractor cannot parse, and the SciFact-tuned NLI head was not trained on technical CS or math claims.
- Multi-column scientific papers with tables that span columns. Reading order gets confused and claims come out malformed.
- Non-biomedical papers (finance, policy, law, ML/CS). The system will run, but the verdicts should not be trusted. See `CHALLENGES.md`.
- Papers with image-only figures referenced in claims. The tool reads text, not images.

If you want to evaluate the project, the cleanest test is a recent open-access PMC review article plus 3 to 5 of its cited primary studies. Upload the review as the draft, the primary studies as the cited papers. That is the workflow this tool was tuned for.

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

The whole pipeline runs locally. No external API calls happen during classification. The only optional request-time outside call is when you click "Explain this verdict" on a single claim, which sends just that claim and its evidence to a language model for a one paragraph plain English explanation. The server also makes a single TLS handshake to the LLM endpoint at startup (a macOS-specific warmup that is a no-op on Linux production); no real data is sent.

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

The system has been tuned and tested on a private collection of biomedical fact verification cases covering all four verdict types. Specific accuracy numbers are not published in this README because the eval harness needs a rewire before the numbers can be reproduced from this repo. See [`CHALLENGES.md`](CHALLENGES.md) for an honest account of what works and what doesn't.

Speed depends on hardware. On a shared CPU server (the developer environment), a typical run takes 3 to 18 minutes for one paper. On a workstation with more cores it is faster. Re-uploading the same PDF is much faster because the embeddings are cached on disk.

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

For the architecture and design decisions: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). For why weak evidence always gets marked insufficient instead of supported: [`docs/TRUST_MODEL.md`](docs/TRUST_MODEL.md). For the full list of models and what each one does: [`docs/MODELS_AND_TERMS.md`](docs/MODELS_AND_TERMS.md). For known limitations: [`CHALLENGES.md`](CHALLENGES.md). For model and dataset attribution: [`ATTRIBUTIONS.md`](ATTRIBUTIONS.md).

## License

MIT. See [LICENSE](LICENSE).
