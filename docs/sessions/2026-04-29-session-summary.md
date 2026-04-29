# EvidenceEngine — Session 2026-04-28 → 04-29

Self-contained brief. Captures what shipped, what was learned, what's open.

## TL;DR

P0 from prior kickoff doc (encoder swap for speed) was killed — premise was outdated. Real prod measurement: 41-171s, not 15-25 min. Pivoted to higher-ROI fixes: dense encoder prewarm, smarter sampling refill, progress UX, extraction filter tightening. All 4 deployed. Visual /browse on prod found 60% false-positive rate on supported claims (methods boilerplate + biblio leaks). Filter fix shipped to address that. **One open concern: retrieval-stage wall time regressed from ~80s to ~12 min on same content after the 2nd deploy — likely shared-CPU noise but worth re-testing next session.**

## What was shipped (6 commits, all local — user needs to push)

| # | Commit | Change | Status |
|---|---|---|---|
| 1 | `fca09ec` | perf(prewarm): BGE dense encoder prewarm in lifespan | ✅ deployed, logs confirm "Dense encoder prewarmed in 41.0s" |
| 2 | `107686d` | fix(extraction): refill 25-cap when filter drops claims | ✅ deployed |
| 3 | `1469a65` | feat(ux): per-stage progress bar + ETA | ✅ deployed |
| 4 | `960a7ed` | fix(extraction): journal-abbrev biblio + methods-procedural | ✅ deployed |
| 5 | `0ea26a9` | docs(lessons): 70 (verify-premise) + 71 (visual /browse mandatory) | ✅ |
| 6 | (pending: this handoff doc) | docs(session): wrap-up | ⏳ |

**User action:** push.
```bash
cd /Users/karankumar/Desktop/vibebuild
git push origin main
```

## Self-critique passes that mattered tonight

1. **Encoder swap pause** — first measurement (153s) flagged the kickoff "15-25 min" claim as stale. Did NOT execute the swap. Saved ~50min of A/B benchmark + likely 5-8pp accuracy regression.
2. **API count vs visual reality** — claimed "8 supported / 17 insufficient looks healthy" too early. /browse via subagent found 6/10 supported = methods/biblio boilerplate. Lesson 71 added.
3. **Filter scope** — initial methods regex too aggressive. Added clinical-noun escape hatch so legit findings ("trial reduced mortality") survive. Smoke test caught the over-rejection.

## Visual QA findings (cardio + multi-doc)

10 supported claims inspected pre-filter:
- 4 cardio = methods boilerplate (rejected by new filter)
- 2 multi-doc = bibliography ref-list ending in journal abbrev (rejected by new filter)
- 4 cardio = arguably real (FDA approval, KDIGO guidelines)

Post-filter retest: same 8 supported count on cardio. Visual QA agent re-running to verify whether the NEW 8 supported are real or new FP shapes (in flight at session end).

## Performance numbers (truth, not handoff doc claim)

Server: shared-cpu-2x:4096MB, sjc, single machine, suspended-when-idle.

| Run | Type | Wall |
|---|---|---|
| BGE multi-doc semaglutide combo | cached | 153s |
| BGE single-doc cardio | fresh, model cold | 171s |
| BGE single-doc infect | fresh, model warm | 41s |
| BGE single-doc onco | fresh, model warm | 82s |
| BGE single-doc dapa-afib (post-prewarm deploy) | fresh | 396s ⚠ |
| BGE multi-doc retest (post-prewarm deploy) | cached | 293s ⚠ |
| BGE single-doc cardio (post-filter deploy) | fresh | 797s ⚠⚠ |

Trend: every deploy seems to slow re-runs. Either (a) Fly machine throttling intensifies, (b) something in code path adds latency. Extraction itself = 1.5s (logged). Retrieval = the long pole. Expected ~80s, observed 12min. Investigate next session.

## Filters shipped (claim_extractor.py)

```python
_BIBLIOGRAPHY_RE  # extended w/ "(ACRONYM). Title-Case Journal-Abbrev." tail
_METHODS_PROCEDURAL_RE  # was/were + assessed/used/considered/...
_CLINICAL_NOUN_RE  # escape hatch: patient/trial/dose/outcome/risk/...
```

Rejected in `_is_likely_claim`: methods sentence with no clinical noun → drop.

## Open priorities for next session

### 🔴 P0 — Investigate post-deploy retrieval slowdown

Retrieval stage went from ~80s typical to 12min on same content after 2nd deploy. Either:
- Fly noisy-neighbor (transient — re-test after 30+ min idle)
- BM25/dense cache invalidation per deploy
- Smart sampling produces different claim set → cache miss on /data dense embeddings (text-keyed)

**Action:** add per-stage timing logs to pipeline orchestrator, re-run cardio fixture in a quiet window.

### 🟡 P1 — Filter refinement (FP 37.5% → target <20%)

Visual QA confirmed filter dropped FP from 50% → 37.5% on cardio retest:
- ✅ "Open Meta-Analyst was used" / tool-name family eliminated
- ✅ One additional real claim surfaced (ADA recommendation)
- ❌ "X were strictly reserved" leaks (adverb between aux + verb breaks regex adjacency)
- ❌ "quality... was assessed using" leaks (escapes via clinical_noun=trials)
- ❌ "p value... was considered" still appeared in dashboard (local test rejects — possible deploy/timing edge)

Refinements:
1. Allow optional adverb: `(?:was|were|are|is)\s+(?:\w+\s+)?(?:assessed|used|considered|...)`
2. Add `was assessed using` as direct procedural pattern
3. Strengthen escape: require BOTH clinical noun AND a finding verb (showed/demonstrated/reduced/increased) before bypassing methods filter

### 🟡 P1 — Self-verify badge UX

Lesson 71 actionable #3: dashboard shows bold green "Supported 80%" even when claim==evidence. Misleading. Change to "Self-supported" or grey when only same-doc evidence + cap badge.

### 🟢 P2 — Multi-domain matrix retry

Earlier matrix bash failed (multipart `-F` arg expansion). Need to retry with proper escaping or use a Python driver. 5 fixtures across cardio/onco/infect/multi-doc/out-of-domain.

### 🟢 P3 — SciFact benchmark re-run

CLAUDE.md priority #2. Numbers stale post all the recent filter changes. Run `eval/run_nli_benchmark.py` via fly ssh.

## What was NOT done (locked or skipped)

| Item | Reason |
|---|---|
| Encoder swap (BGE→MiniLM) | P0 premise outdated — kickoff doc cited 15-25 min, real = 41-171s. Swap saves <30%, risks 5-8pp recall. Killed. |
| Fly machine size bump | User budget cap |
| Multi-doc cite-validated test | Agent's PMC triples were topical not cited. Used as single-doc self-verify instead. |

## Test fixtures inventory (additions tonight)

`/tmp/eevb-bio-fresh/`:
- cardiology/{review-sglt2-mortality, trial-dapa-afib-ablation, trial-empagliflozin-revascularization}.pdf
- infectious-disease/{review-mrna-vaccines, trial-bnt162b2-igg-response, trial-mrna-temporal-dynamics}.pdf
- oncology/{review-dlbcl-cart, trial-tisacel-3yr-update, trial-tisacel-realworld-ball}.pdf

Note: "topical alignment" not "cited". Use as single-doc self-verify, not multi-doc cite-resolver tests.

## Honest concerns to raise next session

1. **Speed regression** is the big unknown. Could be Fly noise, could be something we shipped. Need per-stage timing.
2. **60% FP rate on supported** is the bigger systemic problem than speed. Even if filter works on cardio, every new bio sub-domain may surface new FP shapes (lesson 68 family).
3. **Smart sampling refill** runs `_is_likely_claim` 3-4x per claim. Mild cost, but on 955-candidate docs that adds up. Worth profiling.
4. **Self-verify dominating** — most "supported" verdicts are claim==evidence on the same doc. Real multi-doc cite-resolved support is rare. Either accept that as the bio reality (review papers self-cite a lot) or surface it more honestly in UI.

## Session-bookmarked claude-mem observations

- 6486: Dense encoder not prewarmed despite fast cold-start performance (resolved by tonight's commit)
- 6496: Per-stage progress tracking with ETA added
- Earlier 6024: NLI prewarm "configured but not observed in production startup logs" — verified working tonight ("NLI model (DeBERTa) prewarmed in 33.0s")

## Recommended first move next session

1. Read this doc + `tasks/lessons.md` 70-71.
2. Verify prod health: `curl -sS https://evidenceengine.fly.dev/healthz`.
3. Re-run cardio fixture in quiet window. If <5min → tonight's slowdown was Fly noise. If still >10min → instrument per-stage timing.
4. Fetch visual QA agent output if not yet read: `/private/tmp/claude-501/.../tasks/a86bf3a5dfc4a0c1f.output`.
5. Pick up P1 self-verify badge OR P0 speed investigation depending on (3).
