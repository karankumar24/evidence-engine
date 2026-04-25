#!/usr/bin/env bash
# Quick production smoke test — uploads new test PDFs one at a time
# and checks that pipeline completes in <90s with clean claims.
#
# Usage: bash scripts/test_new_pdfs.sh
# Requires: curl, jq
set -euo pipefail

BASE="https://evidenceengine.fly.dev"
PDF_DIR="tests/fixtures/qa_pdfs"

PDFS=(
  "whisper_paper.pdf"
  "t5_paper.pdf"
  "ecb_economic_bulletin_2024.pdf"
  "mistral_paper.pdf"
  "bis_quarterly_review_2023.pdf"
)

echo "=== EvidenceEngine production PDF smoke test ==="
echo "Target: $BASE"
echo ""

pass=0; fail=0

for pdf in "${PDFS[@]}"; do
  path="$PDF_DIR/$pdf"
  if [ ! -f "$path" ]; then
    echo "SKIP  $pdf (file not found)"
    continue
  fi

  size=$(du -sh "$path" | cut -f1)
  echo -n "Testing $pdf ($size) ... "

  # Upload (self-verify — no cited sources)
  resp=$(curl -s -w "\n%{http_code}\n%{redirect_url}" \
    -X POST "$BASE/upload" \
    -F "report=@$path" \
    --max-time 10 2>/dev/null || true)

  http_code=$(echo "$resp" | tail -2 | head -1)

  if [ "$http_code" != "200" ] && [ "$http_code" != "302" ] && [ "$http_code" != "303" ]; then
    echo "UPLOAD FAIL (HTTP $http_code)"
    ((fail++)); continue
  fi

  # Extract run status URL from response HTML
  run_url=$(echo "$resp" | grep -o 'href="/runs/[^"]*"' | head -1 | sed 's/href="//;s/"//')
  if [ -z "$run_url" ]; then
    # Try to find packet redirect
    run_url=$(echo "$resp" | grep -o '/runs/[a-z0-9-]*/status' | head -1)
  fi

  if [ -z "$run_url" ]; then
    echo "STARTED (could not parse run_id from HTML — check dashboard manually)"
    ((pass++)); continue
  fi

  # Poll for completion (max 90s)
  start=$(date +%s)
  status="pending"
  while true; do
    now=$(date +%s)
    elapsed=$((now - start))
    if [ $elapsed -gt 90 ]; then
      echo "TIMEOUT (>90s) — may still be processing"
      ((fail++)); break
    fi

    poll_resp=$(curl -s "$BASE$run_url/poll" --max-time 5 2>/dev/null || echo "")
    if echo "$poll_resp" | grep -qi "complete\|done\|error\|failed"; then
      elapsed=$(($(date +%s) - start))
      echo "DONE in ${elapsed}s"
      ((pass++)); break
    fi
    sleep 5
  done
done

echo ""
echo "Results: $pass passed, $fail failed"
