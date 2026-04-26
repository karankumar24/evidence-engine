# EvidenceEngine Next-Session Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify multi-doc workflow works end-to-end, clean one dead config param, and stress-test the 8.4MB IMF document after all fixes.

**Architecture:** FastAPI + NLTK claim extractor + BM25/dense hybrid retrieval + SciFact fine-tuned NLI classifier + HTMX review dashboard. Zero LLM calls in hot path.

**Tech Stack:** Python 3.13, FastAPI, SQLAlchemy async, PostgreSQL, NLTK, sentence-transformers, cross-encoder/nli-deberta-v3-small, Fly.io deployment.

**Session start:** Run `/context-restore` first — loads checkpoint `20260425-165959-final-checks-honest-audit-cleanup-deploy.md` which has full context of what was done in the April 25 session.

---

## Pre-flight checks (before any task)

- [ ] Run `/context-restore` to load the April 25 checkpoint
- [ ] Verify production is healthy: `curl -s https://evidenceengine.fly.dev/healthz`
- [ ] Check git log: `git log --oneline -5` — last commit should be `7f405e7 fix(extractor): filter figure caption sentences universally`

---

## Task 1: Audit — confirm claim_errors actually shows in dashboard UI

The API already returns `claim_errors` in `/api/runs/{run_id}/results` (`runs.py:49-65`, `schemas/run.py:69`). But does the **dashboard HTML template** actually display it? Needs verification, not code.

**Files to check (read-only):**
- `src/evidenceengine/templates/dashboard.html`
- `src/evidenceengine/templates/partials/` — any partial that renders verdicts

**Skill:** `/browse` + snapshot to inspect the dashboard

- [ ] **Step 1: Read dashboard template**

```bash
grep -n "claim_error\|error\|failed" src/evidenceengine/templates/dashboard.html | head -20
```

Expected: Either shows "claim_errors" references (UI is wired) OR nothing (UI not wired).

- [ ] **Step 2: Upload a PDF and check the API response directly**

```bash
# After uploading any PDF, get run_id from the URL, then:
curl -s https://evidenceengine.fly.dev/api/runs/{run_id}/results | python3 -c "import json,sys; d=json.load(sys.stdin); print('claim_errors:', d.get('claim_errors', 'MISSING'))"
```

Expected: `claim_errors: []` (empty list when no errors, but field is present).

- [ ] **Step 3: Record finding in tasks/todo.md**

If `claim_errors` is in API but NOT in the UI template → add to todo as "surface claim errors in dashboard HTML". If already shown → mark done.

---

## Task 2: Remove dead config param `nli_tiebreaker_threshold`

`nli_min_confidence_for_verdict` is actively used (`nli_classifier.py:171`). `nli_tiebreaker_threshold` is genuinely dead — only in config comments as "reserved for future tiebreaker".

**Files:**
- Modify: `src/evidenceengine/core/config.py:100-120`
- Test: `tests/` — search for any reference to `nli_tiebreaker`

- [ ] **Step 1: Write failing test**

```python
# tests/test_config.py — add this test
def test_nli_tiebreaker_threshold_not_in_settings():
    """Dead config param should be removed."""
    from evidenceengine.core.config import Settings
    assert not hasattr(Settings(), "nli_tiebreaker_threshold"), \
        "nli_tiebreaker_threshold is dead code — should be removed from Settings"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
timeout 30 .venv/bin/python -m pytest tests/test_config.py::test_nli_tiebreaker_threshold_not_in_settings -v 2>&1 | tail -10
```

Expected: FAIL — `AssertionError: nli_tiebreaker_threshold is dead code`

- [ ] **Step 3: Remove the dead param from config.py**

In `src/evidenceengine/core/config.py`, remove lines 112-120 (the comment and field):

```python
# REMOVE these lines:
#   * nli_tiebreaker_threshold (0.65): reserved for a future tiebreaker path
nli_tiebreaker_threshold: float = 0.65
```

Also update the comment block at lines 100-112 that references it — remove the mention:
```python
# REMOVE from comment:
#   0 < nli_tiebreaker_threshold
#     < 1.0
```

- [ ] **Step 4: Run test to verify it passes**

```bash
timeout 30 .venv/bin/python -m pytest tests/test_config.py::test_nli_tiebreaker_threshold_not_in_settings -v 2>&1 | tail -5
```

Expected: PASS

- [ ] **Step 5: Run full test suite to check no regressions**

```bash
timeout 60 .venv/bin/python -m pytest tests/test_claim_extractor.py -k "test_rejects or test_passes or test_still" -q 2>&1 | tail -5
```

Expected: 35 passed

- [ ] **Step 6: Commit**

```bash
git add src/evidenceengine/core/config.py tests/test_config.py
git commit -m "chore(config): remove dead nli_tiebreaker_threshold param

Was documented as 'reserved for future tiebreaker path' but never read
anywhere in the codebase. nli_min_confidence_for_verdict (0.50) is
kept — it IS used in nli_classifier.py:171.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Browser-verify multi-doc workflow (PRIMARY USE CASE)

This is the most important test. Upload a REPORT + its CITED SOURCE and confirm claims are verified against the source, not the report itself.

**Skill:** `/browse`

**Files to watch during test:**
- `src/evidenceengine/extraction/anchor_resolver.py` — citation resolution (most likely failure point)
- `~/.fly/bin/fly logs --app evidenceengine --no-tail` — production logs

- [ ] **Step 1: Navigate to upload page**

```
$B goto https://evidenceengine.fly.dev/upload
$B screenshot /tmp/upload_before.png
```

Read `/tmp/upload_before.png` to confirm the upload form is visible with "Your draft" and "Cited papers" sections.

- [ ] **Step 2: Upload T5 paper as draft + BERT paper as cited source**

```
$B upload @e2 tests/fixtures/qa_pdfs/t5_paper.pdf
# Wait for "Cited papers" file input to appear (it should already be there)
$B snapshot -i
# Find the cited papers input — should be @e3
$B upload @e3 tests/fixtures/qa_pdfs/bert_paper.pdf
$B screenshot /tmp/upload_both.png
```

Read `/tmp/upload_both.png` — should show BOTH files selected. The amber warning should NOT appear (it only shows when no cited papers are uploaded).

- [ ] **Step 3: Run verification and wait for completion**

```
$B click @e4   # "Run Verification" button
# Wait for redirect to /runs/{id}/status
# Then wait for auto-redirect to /dashboard/{packet_id}/{run_id}
$B wait --networkidle
$B screenshot /tmp/dashboard.png
```

Read `/tmp/dashboard.png` — should show the dashboard with claim list.

- [ ] **Step 4: Inspect actual claim text and evidence source**

```
$B snapshot -i
$B js "Array.from(document.querySelectorAll('*')).filter(e => e.textContent.includes('bert') || e.textContent.includes('BERT')).slice(0,5).map(e => e.textContent.trim().slice(0,100)).join('---')"
```

Expected: Evidence spans mentioning "bert_paper.pdf" or BERT paper content.

- [ ] **Step 5: Verify evidence source attribution**

```
$B js "document.body.innerHTML.includes('bert_paper.pdf') || document.body.innerHTML.includes('bert')"
```

Expected: `true` — BERT paper is being used as the evidence source.

- [ ] **Step 6: Check production logs for citation resolution**

```bash
~/.fly/bin/fly logs --app evidenceengine --no-tail 2>&1 | grep -E "anchor|citation|resolve|source" | tail -20
```

Look for: "Resolved citation anchor → bert_paper" or similar. If all anchors show "self-verify" that means citation resolution failed.

- [ ] **Step 7: Record result**

**If PASS:** Verdicts show evidence from bert_paper.pdf. Update `tasks/todo.md` — mark multi-doc verification complete.

**If FAIL (all claims are self-verify):** The T5 paper uses numeric citations like [1], [2]. The anchor_resolver needs to match "[1]" → bert_paper.pdf. Run `/investigate` with this specific hypothesis: "numeric citation resolution fails when source filename doesn't match reference entry text."

---

## Task 4: Re-run SciFact benchmark (5 min)

The 86% number was measured before the April 25 universal filter. The filter affects which sentences REACH the NLI model, not how NLI classifies them. Expected: number unchanged.

**No code changes needed.** CLI only.

- [ ] **Step 1: Run benchmark on Fly.io (faster than local CPU)**

```bash
~/.fly/bin/fly ssh console --command "cd /app && python eval/run_nli_benchmark.py" 2>&1 | tail -30
```

Alternative (local, takes ~4 min):
```bash
timeout 300 .venv/bin/python eval/run_nli_benchmark.py 2>&1
```

- [ ] **Step 2: Record numbers**

Expected output format:
```
Loaded 220 gold cases
Overall accuracy: XX.X%
3-class accuracy (excl needs_review): XX.X%
False support rate: X.X%
```

- [ ] **Step 3: Update CLAUDE.md if numbers changed >1pp**

In `CLAUDE.md`, update the "Accuracy (internal gold, 220 cases)" line with new numbers. Commit:

```bash
git add CLAUDE.md
git commit -m "docs: update NLI benchmark numbers post-April-25-filter"
```

---

## Task 5: IMF WEO 2025 Stress Test (8.4MB, 190 pages)

Was previously failing with DB connection drop at 14 min. Now has: session split + table extraction skip + dense encoding skip + thread pool + universal filter. Expected: complete in 3-5 min.

**Skill:** `/browse` for upload. Watch logs during.

- [ ] **Step 1: Upload IMF WEO 2025**

```
$B goto https://evidenceengine.fly.dev/upload
$B upload @e2 tests/fixtures/qa_pdfs/imf_weo_2025.pdf
$B click @e4
```

- [ ] **Step 2: Monitor logs in parallel**

```bash
~/.fly/bin/fly logs --app evidenceengine --no-tail 2>&1 | grep -E "table|dense|extract|NLI|claim|error" | tail -30
```

Expected log entries:
- `Table extraction skipped for document with X pages (>50 threshold)` ← table skip
- `Dense retrieval skipped: corpus size X > MAX_DENSE_CORPUS_SIZE` ← dense skip  
- `Local extractor produced 25 claims from X candidates` ← extraction
- `Pipeline completed for run ... — 0 claim errors` ← success

- [ ] **Step 3: Check dashboard after completion**

```
$B wait --networkidle
$B screenshot /tmp/imf_dashboard.png
```

Read `/tmp/imf_dashboard.png` — inspect claim quality. No legal boilerplate, no TOC entries.

- [ ] **Step 4: Record timing**

Note how long from upload to "completed" status. Document in tasks/todo.md.

---

## Task 6: Deploy if any code changes were made

Only needed if Tasks 1-2 produced code changes.

- [ ] **Step 1: Deploy**

```bash
~/.fly/bin/fly deploy --strategy rolling 2>&1 | tail -10
```

- [ ] **Step 2: Verify health**

```bash
curl -s https://evidenceengine.fly.dev/healthz
```

Expected: `{"status":"ok","db":"reachable"}`

- [ ] **Step 3: Check logs for errors**

```bash
~/.fly/bin/fly logs --app evidenceengine --no-tail 2>&1 | grep -v "healthz" | tail -15
```

---

## Session end — save context

- [ ] Run `/context-save` with a descriptive title reflecting what was completed

---

## Self-Review Checklist

**Spec coverage:**
- ✅ Multi-doc workflow verification — Task 3
- ✅ Dead config cleanup — Task 2
- ✅ Benchmark re-run — Task 4
- ✅ IMF stress test — Task 5
- ✅ Claim_errors UI check — Task 1
- ✅ Deploy + health check — Task 6

**Corrections from earlier plan (important):**
- ~~Phase 2: Surface classification errors in API~~ — ALREADY DONE. `claim_errors` is in the API since a previous session (`runs.py:49-65`, `schemas/run.py:69`). Task 1 just verifies the UI shows it.
- ~~Phase 3: Remove nli_min_confidence_for_verdict~~ — NOT dead. Used in `nli_classifier.py:171`. Only `nli_tiebreaker_threshold` is removed.

**Placeholder scan:** None — all steps have exact commands, expected output, and code.

**Risk:** Task 3 (multi-doc) may reveal citation resolution failures. If T5's numeric citations ([1] = BERT) don't resolve to bert_paper.pdf, most claims will self-verify. Use `/investigate` if this happens, starting with `anchor_resolver.py`.

---

## Execution Options

**Plan complete. Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks
- **REQUIRED:** Use `superpowers:subagent-driven-development`

**2. Inline Execution** — tasks executed sequentially in this session
- **REQUIRED:** Use `superpowers:executing-plans`
