"""CI quality gate: fail if false_support_rate exceeds threshold.

Usage:
    python eval/ci_gate.py

Environment variables:
    EVAL_FALSE_SUPPORT_THRESHOLD  (float, default 0.10)
    OPENAI_API_KEY                (required for classify_claim)
    EVAL_CONCURRENCY              (int, default 5)

Exit codes:
    0 — gate passed (false_support_rate <= threshold)
    1 — gate failed (false_support_rate > threshold)
    2 — benchmark load error or evaluation error
"""

import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is on path when run as a script (python eval/ci_gate.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.runner import run_evaluation

DEFAULT_THRESHOLD = 0.10


def main() -> int:
    threshold_str = os.environ.get("EVAL_FALSE_SUPPORT_THRESHOLD", str(DEFAULT_THRESHOLD))
    try:
        threshold = float(threshold_str)
    except ValueError:
        print(
            f"ERROR: EVAL_FALSE_SUPPORT_THRESHOLD='{threshold_str}' is not a valid float",
            file=sys.stderr,
        )
        return 2

    concurrency = int(os.environ.get("EVAL_CONCURRENCY", "5"))

    print(f"Running EvidenceEngine eval gate (threshold: false_support_rate <= {threshold})")

    try:
        report, results = asyncio.run(run_evaluation(concurrency=concurrency))
    except Exception as exc:
        print(f"ERROR: Evaluation failed: {exc}", file=sys.stderr)
        return 2

    n_gold = len([r for r in results if r.source == "gold"])
    print(f"Evaluated {n_gold} gold cases")
    print(f"  false_support_rate         = {report.false_support_rate:.4f}  (threshold: {threshold})")
    print(f"  contradiction_recall       = {report.contradiction_recall:.4f}")
    print(f"  insufficient_support_prec  = {report.insufficient_support_precision:.4f}")
    print(f"  evidence_retrieval_recall  = {report.evidence_retrieval_recall:.4f}")

    if report.false_support_rate > threshold:
        print(
            f"\nGATE FAILED: false_support_rate {report.false_support_rate:.4f} "
            f"> threshold {threshold}",
            file=sys.stderr,
        )
        return 1

    print(f"\nGATE PASSED: false_support_rate {report.false_support_rate:.4f} <= {threshold}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
