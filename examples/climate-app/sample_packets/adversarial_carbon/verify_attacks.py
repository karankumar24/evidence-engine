"""Live verification of the four adversarial attacks against EvidenceEngine.

Uploads the adversarial_carbon packet to a running EvidenceEngine deployment
(local or Fly), polls for completion, then asserts each attack reached the
expected verdict. Complements tests/test_adversarial_pipeline.py (which uses
mocked LLMs) with a real end-to-end run against live models.

Requires:
  - A running EvidenceEngine API (default: http://localhost:8000, override via
    EE_BASE_URL env var).
  - Packet PDFs generated from the .txt files in this directory — run
    `tools/txt_to_pdf.py` first if needed.
  - Free-tier LLM quota available (~15 requests minimum for 4 attacks + 1 control).

Exit codes:
  0 — all four attacks + control passed
  1 — at least one attack's verdict did not match expected
  2 — infrastructure error (upload failed, run didn't complete, etc.)

Run: python examples/climate-app/sample_packets/adversarial_carbon/verify_attacks.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    print("httpx required — install with: uv pip install httpx", file=sys.stderr)
    sys.exit(2)


HERE = Path(__file__).parent
BASE_URL = os.environ.get("EE_BASE_URL", "http://localhost:8000")
POLL_INTERVAL = 10
MAX_WAIT_SECONDS = 900


def main() -> int:
    meta = json.loads((HERE / "metadata.json").read_text())
    report_pdf = HERE / meta["report_file"].replace(".txt", ".pdf")
    source_pdfs = [HERE / s.replace(".txt", ".pdf") for s in meta["source_files"]]

    for p in [report_pdf] + source_pdfs:
        if not p.exists():
            print(f"missing {p} — run tools/txt_to_pdf.py first", file=sys.stderr)
            return 2

    client = httpx.Client(base_url=BASE_URL, timeout=60)

    # 1. Upload
    files = [("report", (report_pdf.name, report_pdf.read_bytes(), "application/pdf"))]
    files.extend(
        ("sources", (p.name, p.read_bytes(), "application/pdf")) for p in source_pdfs
    )
    upload = client.post("/upload", files=files, follow_redirects=False)
    if upload.status_code not in (200, 302, 303):
        print(f"upload failed: {upload.status_code} {upload.text[:200]}", file=sys.stderr)
        return 2
    location = upload.headers.get("location", "")
    run_id = location.split("/runs/")[-1].split("/")[0] if "/runs/" in location else None
    if not run_id:
        print(f"could not parse run_id from location={location!r}", file=sys.stderr)
        return 2
    print(f"uploaded — run_id={run_id}")

    # 2. Poll
    deadline = time.time() + MAX_WAIT_SECONDS
    while time.time() < deadline:
        r = client.get(f"/api/runs/{run_id}")
        status = r.json().get("status")
        print(f"[{int(time.time() - deadline + MAX_WAIT_SECONDS)}s] status={status}")
        if status in ("completed", "failed"):
            break
        time.sleep(POLL_INTERVAL)
    else:
        print("timed out waiting for run", file=sys.stderr)
        return 2

    if status != "completed":
        print(f"run ended with status={status}", file=sys.stderr)
        return 2

    # 3. Fetch results
    results = client.get(f"/api/runs/{run_id}/results").json()
    verdicts = results.get("verdicts", [])
    if not verdicts:
        print("no verdicts returned", file=sys.stderr)
        return 2

    # 4. Assert each attack (matched by fragment of claim_text)
    expectations = [
        ("methane concentrations declined", ["needs_review"], "Attack 1: fabricated citation"),
        ("decreased 12 percent", ["contradicted"], "Attack 2: directly contradicted"),
        ("record high", ["insufficient_support", "needs_review"], "Attack 3: cherry-picked scope"),
        ("Land-use change emissions averaged", ["insufficient_support", "needs_review"], "Attack 4: self-verify trap"),
        ("reached approximately 36.6", ["supported"], "Control: should be supported"),
    ]

    # Load claims to map claim_id -> claim_text
    claims = {c["id"]: c for c in results.get("claims", [])}

    failures = 0
    print()
    print("=" * 64)
    print("ADVERSARIAL ATTACK VERIFICATION")
    print("=" * 64)
    for fragment, expected_verdicts, label in expectations:
        matched_verdicts = []
        for v in verdicts:
            claim = claims.get(v["claim_id"], {})
            if fragment.lower() in claim.get("claim_text", "").lower():
                matched_verdicts.append(v)
        if not matched_verdicts:
            print(f"  ⚠  {label}: NO CLAIM MATCHED fragment={fragment!r}")
            failures += 1
            continue
        v = matched_verdicts[0]
        actual = v["verdict_type"]
        ok = actual in expected_verdicts
        mark = "✓" if ok else "✗"
        print(f"  {mark}  {label}")
        print(f"      claim={claim.get('claim_text', '')[:100]!r}")
        print(f"      expected={expected_verdicts}  got={actual}  conf={v.get('confidence_score')}")
        print(f"      reasoning={(v.get('reasoning') or '')[:160]!r}")
        if not ok:
            failures += 1

    # 5. Telemetry sanity
    telemetry = results.get("pipeline_config", {}).get("classification_telemetry", {})
    if telemetry:
        print()
        print("telemetry:")
        for k, v in telemetry.items():
            print(f"  {k}: {v}")

    print()
    if failures == 0:
        print(f"All {len(expectations)} attacks PASSED — trust model holds.")
        return 0
    print(f"{failures} of {len(expectations)} attacks FAILED — trust model has a leak.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
