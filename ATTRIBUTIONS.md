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
| **SciFact** (Wadden et al., 2020) | Fine-tuning the NLI classifier | **CC BY-NC 2.0** | The NC ("non-commercial") clause is real. The fine-tuned weights derived from SciFact inherit this restriction in the eyes of most lawyers. **If you want to use the SciFact-fine-tuned checkpoint commercially, talk to a lawyer or train your own checkpoint on a license-clean dataset.** EvidenceEngine itself is MIT but the *fine-tuned model file* is the load-bearing part of the verdict step. |

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

## Datasets used in development eval (not shipped)

The internal 220-case gold benchmark referenced in commit history is private and is not redistributed in this repo. SciFact dev set (CC BY-NC 2.0) is used for offline NLI calibration and is not included in the repo either.

## Logo / images

Hero screenshot in older README revisions was removed pending an honest re-capture from a real multi-document run.

## What you need to do if you redistribute

1. Keep this file with the project.
2. If you ship the SciFact-fine-tuned checkpoint, surface the CC BY-NC restriction to your users.
3. If you keep PyMuPDF, plan for AGPL compliance on any hosted deployment, or replace it.
4. If you train your own NLI checkpoint on a permissive dataset, you can drop the SciFact note and ship freely under MIT.
