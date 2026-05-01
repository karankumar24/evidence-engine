# Challenges and Limitations

EvidenceEngine is a working portfolio piece, not a finished product. This file lists what the system does NOT handle well, the trade-offs that were made, and the things a future maintainer (or you) would need to fix to ship it commercially. No marketing voice. Read before relying on it.

## Domain narrowness

The verdict classifier is a `cross-encoder/nli-deberta-v3-small` fine-tuned on SciFact, a biomedical fact verification dataset. On in-domain biomedical claims (clinical trials, epidemiology, drug efficacy, basic life sciences) it performs reasonably. On other domains the same model degrades 15 to 25 percentage points based on internal smoke tests:

- Machine learning / computer science papers (math, model architecture, benchmark claims)
- Finance / economics (numeric claims with mixed temporal context)
- Policy and law
- Climate science (mixed; close to bio but not identical claim shapes)

Use it on a non-biomedical paper and the verdicts will look authoritative without being right. The system does not currently detect or warn that an uploaded paper is out-of-domain.

## Sampling cap and what it costs

A run extracts at most **25 claims** (`max_claims_absolute = 25` in `core/config.py`), capped further by `max_claims_per_page = 4` for short papers. This is a hard cap to keep the pipeline finishable on a shared CPU within a few minutes.

Consequences:
- A 30-page review with 200 candidate claims gets a 25-claim sample.
- Sampling is biased toward sentences with citation markers (Vancouver `[1]`, APA `(Smith et al., 2023)`). This was a deliberate choice for the "verify cited claims" use case, but it means **novel uncited claims (the author's own contribution) rarely get verified**.
- If your real failure mode is "did I overstate my own findings," EvidenceEngine helps less than you would hope.

## Self-verify mode caveats

When no cited papers are uploaded (Mode A), the system retrieves evidence from other paragraphs of the *same* paper. This is useful for catching internal contradictions and unsupported assertions, but it is not a substitute for verifying citations.

Implementation specifics:
- The 80% confidence cap is applied **only to `supported` verdicts** (see `classification/pipeline.py` self-verify guard). `contradicted` and `insufficient_support` verdicts are not capped, on the reasoning that an internal contradiction is meaningful even when the source is the same document.
- Detection of self-verify mode is heuristic: every retrieved evidence span must come from the same `source_document_id` as the claim. A multi-doc upload where the retriever happened to surface only one doc's spans would also trigger the cap. This is a feature, not a bug, but worth knowing.

## Retrieval ceiling

The hybrid BM25 + BGE-base dense + ms-marco reranker pipeline tops out at the top-N evidence chunks per claim (see `core/config.py` `evidence_top_k`). If the cited paper's PDF text is corrupted (scanned image, multi-column tables, math-heavy LaTeX) the retriever cannot find passages it cannot read. PyMuPDF parses most modern PDFs well but fails silently on older or scanned scans. There is no OCR pass.

## Anchor resolution

Claims are mapped back to specific cited papers via the citation anchor inside the claim sentence. The resolver:
- Handles Vancouver-style `[1]`, `[1,2,5]`, `[1-3]` reliably.
- Handles APA `(Smith et al., 2023)` when the matched first author + year exists in the parsed reference list.
- Fails on inline numeric references that don't match a reference list entry, on superscript markers that PyMuPDF does not extract, and on any reference list that is itself malformed (citations that span pages, OCR'd, or in a non-standard format).

When anchor resolution fails, the claim falls back to broad multi-document retrieval (Mode B3) and gets a `needs_review` verdict if the evidence is weak.

## Eval reproducibility

The internal benchmark numbers that previously appeared in the README (e.g. "84.8% 3-class accuracy") were measured against a private 220-case gold set. **The repo does not currently include a working eval harness that reproduces those numbers from scratch.** Specifically:
- `eval/runner.py` imports `evidenceengine.classification.classifier`, which was deleted during the Phase A refactor. The runner needs a rewire.
- `eval/results/` is gitignored.
- The gold set itself contains private data and is not part of the repo.

What this means: **do not trust the historical numbers as reproducible from this codebase as-is.** The metrics belong in CHALLENGES.md (here, honestly) until the eval harness is rewired and a result JSON is committed.

## PDF parsing failures

PyMuPDF handles most PDFs born digital. Things it does badly:
- Scanned image PDFs (no text layer): zero extraction. No OCR fallback shipped.
- PDFs with tables that span columns: text reading order is wrong, claims may be malformed.
- Math-heavy papers (LaTeX rendered to glyph soup): formula text is often unreadable.
- Old PDFs with subset / non-Unicode fonts: character substitution makes claims look like mojibake.

The system does not currently surface any of this to the user. A claim that reads as gibberish was probably extracted from a broken PDF.

## Public demo posture

There is no public demo right now. It was scaled to zero before public release of the repo because the upload endpoint and the on-demand "Explain this verdict" LLM call were unauthenticated and unrate-limited. Anyone could trigger arbitrary cost.

If you bring it back online, you need to:
- Add per-IP and per-session rate limits (e.g. via `slowapi`).
- Cap concurrent runs.
- Cap total upload size per session.
- Either auth-gate `/api/claims/{id}/explain` or remove it.

The unauthenticated-by-design posture was correct for a single-user dev deployment. It is not correct for a public demo.

## "Explain this verdict" is opt-in for a reason

Clicking the Explain button on a claim sends the claim text and the top retrieved evidence spans to an external LLM (Gemini or OpenAI-compatible, depending on which API key is set). The body of your draft does not leave your server, but the specific claim and the evidence quotes do. Do not click Explain on confidential drafts.

There is no audit log of which claims have been Explained. Adding one is a future improvement.

## CI / CD

There isn't any. Every deploy is manual (`fly deploy --strategy rolling`). No automated tests run on push. The `tests/` directory has unit tests but no CI runner is configured.

## Speed

On the developer's shared-cpu-2x Fly machine, a typical run is 3 to 18 minutes depending on PDF size and concurrent tenant load. On a workstation with more cores it is faster. The 18-minute tail is real and it is the largest single complaint from internal testing. The bottleneck is NLI inference: 25 claims * top-N evidence spans * one cross-encoder call each, on CPU. A GPU or a smaller distilled NLI head would help; both are out of scope for this iteration.

## Things deferred from this iteration

- ML / CS paper support (would need a different fine-tuned NLI head).
- Diversified claim sampling (cited + uncited mix).
- OCR fallback for scanned PDFs.
- A hosted demo with auth and rate limits.
- Reproducible eval harness with committed result JSON.
- CI on push.
- An audit log for the Explain endpoint.

If you forked this repo and wanted to ship something real, those are the next six tickets.
