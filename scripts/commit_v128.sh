#!/usr/bin/env bash
# Run AFTER closing VS Code / Antigravity — the index contention from their
# background git processes prevents the commit from landing during the session.
set -euo pipefail
cd "$(dirname "$0")/.."

pkill -9 -f "git " 2>/dev/null || true
sleep 2
rm -f .git/index.lock

git add \
  tests/test_adversarial_pipeline.py \
  tests/test_fallback_chain_exhaustion.py \
  tests/test_nli_second_opinion.py \
  examples/climate-app/sample_packets/adversarial_carbon/ \
  src/evidenceengine/classification/nli_second_opinion.py \
  src/evidenceengine/classification/pipeline.py \
  src/evidenceengine/core/config.py \
  eval/metrics/compute.py \
  eval/runner.py \
  eval/reliability.py \
  pyproject.toml \
  tasks/v1.2.8-grade.md \
  tasks/todo.md \
  scripts/commit_v128.sh

git commit -m "phase E+F+G: adversarial tests + chain exhaustion + NLI second-opinion + eval honesty

Final v1.2.8 round. 78 tests passing locally (+12 NLI decision tests).

New trust layers:
  - Local NLI second-opinion (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
    runs after the LLM; when NLI says contradiction at >=0.85 prob but LLM
    said SUPPORTED, verdict is forced to needs_review. One-way downgrade,
    never upgrades. Gracefully fails open if transformers/torch missing.
    Adds deterministic contradiction detection with zero API cost.
  - Telemetry counter nli_second_opinion_overrides exposed via pipeline_config.

Eval honesty:
  - _safe_divide(0, 0) asymmetry fixed: FSR uses vacuous=0.0 (correct for
    lower-is-better ratios), recall/precision keep vacuous=1.0.
  - Runner --resume now retries chain_exhausted and other_error cases
    instead of treating them as done.
  - PersistedCaseResult gained a confidence field (feeds reliability curve).

Reliability diagram utility (eval/reliability.py): reads any JSONL and
produces ASCII calibration bins + ECE + per-verdict confusion matrix.

Adversarial packet + 7 integration tests (fabricated / contradicted /
cherry-picked / self-verify / SAME-DOC tag wire-through / EXTERNAL tag
wire-through / telemetry).

4 fallback-chain exhaustion tests: all-429 raises RuntimeError, chain-retry
triggers on exhaustion, auth-error short-circuits, first-success-wins.

pyproject.toml torch pin now gates on sys_platform=='linux' so local macOS
installs no longer fail on the non-existent torch==2.8.0+cpu mac wheel.

See tasks/v1.2.8-grade.md for the full B- -> A writeup.

Co-Authored-By: Claude <noreply@anthropic.com>"

git log --oneline -6
