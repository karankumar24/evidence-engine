# Attributions

EvidenceEngine ships under MIT (see [LICENSE](LICENSE)) but bundles or downloads several models, datasets, and libraries with their own licenses. This file lists each one and flags any license that needs care if you redistribute the service.

## Models

| Model | Used for | License | Notes |
|---|---|---|---|
| `cross-encoder/nli-deberta-v3-small` | NLI verdict classifier (base + SciFact fine-tuned) | Apache 2.0 (model weights); MIT (DeBERTa code) | Fine-tuned on SciFact; see SciFact note below. |
| `BAAI/bge-base-en-v1.5` | Dense retrieval embedding model | MIT | Pre-downloaded in Dockerfile. |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | Reranker | Apache 2.0 | Pre-downloaded in Dockerfile. |
| `sentence-transformers/all-MiniLM-L6-v2` | Earlier dense retrieval default; not in active hot path on production but referenced in some test fixtures and config defaults | Apache 2.0 | |

## Datasets

| Dataset | Used for | License | Notes |
|---|---|---|---|
| **SciFact** (Wadden et al., 2020) | Fine-tuning the NLI classifier and offline NLI eval | **CC BY-NC 2.0** | The NC ("non-commercial") clause is real. The fine-tuned weights derived from SciFact inherit this restriction in the eyes of most lawyers. **If you want to use the SciFact-fine-tuned checkpoint commercially, talk to a lawyer or train your own checkpoint on a license-clean dataset.** EvidenceEngine itself is MIT but the *fine-tuned model file* is the load-bearing part of the verdict step. The eval scripts download the SciFact archive (`claims_dev.jsonl`, `claims_test.jsonl`, `corpus.jsonl`, etc.) into `eval/data/scifact/`, which is not tracked in git; the SciFact CC BY-NC 2.0 terms apply to those files. |

## Libraries (runtime)

| Library | License | Notes |
|---|---|---|
| FastAPI | MIT | |
| SQLAlchemy | MIT | |
| HTMX | BSD 2-Clause | |
| Alpine.js | MIT | |
| Tailwind CSS | MIT | |
| PostgreSQL | PostgreSQL License (BSD-style) | |
| **PyMuPDF** (fitz) | **AGPL 3.0** | **Copyleft, important.** AGPL has reach into network services. If you fork EvidenceEngine and run a public service, you may be required to publish your full server-side source under AGPL terms. The MIT license on EvidenceEngine code does *not* override PyMuPDF's AGPL on the dependency. If this is a problem for your use case, swap PyMuPDF for a PDF parser with a permissive license (`pdfplumber` is Apache 2.0; quality differs). |
| `bm25s` | MIT | Fast BM25 retrieval. |
| `sentence-transformers` | Apache 2.0 | Wraps the dense + reranker models. |
| `nltk` | Apache 2.0 | Sentence tokenizer for claim extraction. |
| `transformers` (Hugging Face) | Apache 2.0 | Loads the NLI model. |
| `torch` (CPU build) | BSD-style | Pinned to CPU-only in this project to keep the Docker image small. |
| `numpy`, `scipy` | BSD | |
| `alembic` | MIT | DB migrations. |
| `uvicorn` | BSD | ASGI server. |
| `pydantic`, `pydantic-settings` | MIT | Config + request/response models. |
| `asyncpg` | Apache 2.0 | Async Postgres driver. |
| `python-docx` | MIT | DOCX parsing. |
| `python-magic` | MIT | MIME-type detection on upload. |
| `python-multipart` | Apache 2.0 | FastAPI form/file upload support. |
| `openai` (SDK) | Apache 2.0 | Used only by the optional Explain endpoint to call OpenAI-compatible LLM providers. Not used in the verdict hot path. |
| `python-json-logger` | BSD | Structured JSON logs. |
| `datasets` (Hugging Face) | Apache 2.0 | Loads SciFact for eval scripts. |
| `matplotlib` | PSF / matplotlib license (BSD-style) | Plots threshold sweeps in `eval/`. |
| `jinja2` | BSD | Template engine for HTMX views. |

## Datasets used in development eval

The 220-case gold benchmark in `eval/benchmark/fixtures/gold/` (climate and energy claims) is part of this repo. SciFact (CC BY-NC 2.0) is downloaded on demand into `eval/data/scifact/` and is not tracked in git. The Docker image does NOT ship SciFact: `.dockerignore` excludes `eval/data/`, so production deploys do not carry the dataset. If you fork the source repo and redistribute it, the CC BY-NC restriction follows the SciFact files in your tree; if you build and distribute the Docker image as-is, the dataset is not in it.

## Logo / images

Hero screenshot in older README revisions was removed pending an honest re-capture from a real multi-document run.

## What you need to do if you redistribute

1. Keep this file with the project.
2. If you ship the SciFact-fine-tuned checkpoint, surface the CC BY-NC restriction to your users.
3. If you keep PyMuPDF, plan for AGPL compliance on any hosted deployment, or replace it.
4. If you train your own NLI checkpoint on a permissive dataset, you can drop the SciFact note and ship freely under MIT.
