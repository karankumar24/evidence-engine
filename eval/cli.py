"""Standalone CLI for the EvidenceEngine evaluation harness.

Requirements:
- LLM_API_KEY (or legacy OPENAI_API_KEY) env var must be set
- No running FastAPI server is required — runner calls classify_claim() directly
- Runnable as: python -m eval.cli run [options]

Usage:
    python -m eval.cli run [--output PATH] [--concurrency N] [--max-cases N]
                           [--no-resume] [--output-format {json|text}]
    python -m eval.cli run --help

Per-case results are appended to --output (default eval/results/latest.jsonl) as
they complete; --resume (default true) skips cases already present so a crash
or rate-limit pause can pick up where it left off.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from eval.runner import RunSummary, run_evaluation


def _print_text_report(report, results, summary: RunSummary) -> None:
    print("=" * 64)
    print("EvidenceEngine Eval Report")
    print("=" * 64)
    gold_n = sum(1 for r in results if r.source == "gold")
    synth_n = sum(1 for r in results if r.source == "synthetic")
    print(f"Cases evaluated:  gold={gold_n}  synthetic={synth_n}")
    print()
    print(f"Coverage          : {summary.n_classified}/{summary.n_total}  ({summary.coverage:.1%})")
    print(f"  chain_exhausted : {summary.n_chain_exhausted}  (no LLM verdict — rate-limited or 5xx)")
    print(f"  auth_error      : {summary.n_auth_error}  (key/permission issue)")
    print(f"  other_error     : {summary.n_other_error}")
    print(f"  → metrics below are computed ONLY over classified cases")
    print()
    print(f"false_support_rate          : {report.false_support_rate:.4f}  (lower is better)")
    print(f"contradiction_recall        : {report.contradiction_recall:.4f}  (higher is better)")
    print(f"insufficient_support_prec   : {report.insufficient_support_precision:.4f}  (higher is better)")
    print(f"evidence_retrieval_recall   : {report.evidence_retrieval_recall:.4f}  (higher is better)")
    print()
    print("Denominators:")
    print(f"  n_gold                    : {report.n_gold}")
    print(f"  n_gold_contradicted       : {report.n_gold_contradicted}")
    print(f"  n_gold_insufficient       : {report.n_gold_insufficient}")
    print(f"  n_predicted_insufficient  : {report.n_predicted_insufficient}")
    print(f"  n_gold_with_evidence      : {report.n_gold_with_evidence}")
    print()
    print(f"Per-case JSONL: {summary.output_path}")
    print("=" * 64)


def _print_json_report(report, results, summary: RunSummary) -> None:
    output = {
        "metrics": report.model_dump(),
        "summary": summary.model_dump(),
        "summary_extras": {
            "n_gold": sum(1 for r in results if r.source == "gold"),
            "n_synthetic": sum(1 for r in results if r.source == "synthetic"),
        },
    }
    print(json.dumps(output, indent=2))


async def _run_cmd(args: argparse.Namespace) -> int:
    report, results, summary = await run_evaluation(
        output_path=Path(args.output),
        concurrency=args.concurrency,
        resume=not args.no_resume,
        max_cases=args.max_cases,
    )
    if args.output_format == "json":
        _print_json_report(report, results, summary)
    else:
        _print_text_report(report, results, summary)
    # Honest exit: non-zero when coverage is dangerously low so CI surfaces it
    if summary.coverage < 0.50:
        print(
            f"\nWARNING: only {summary.coverage:.1%} of cases were classified — "
            f"metrics may be unreliable. Re-run with `--resume` to pick up where left off.",
            file=sys.stderr,
        )
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval",
        description="EvidenceEngine evaluation harness — no web server required",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full benchmark evaluation")
    run_parser.add_argument(
        "--output",
        default="eval/results/latest.jsonl",
        help="Per-case JSONL output path (default: eval/results/latest.jsonl). "
             "Re-using the same path enables resume.",
    )
    run_parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Max concurrent classify_claim calls (default: 3, friendly to free tier)",
    )
    run_parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit number of cases to evaluate (after resume skip). Useful for "
             "smoke runs before a full 220-case sweep.",
    )
    run_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume — start fresh even if --output file already has results.",
    )
    run_parser.add_argument(
        "--output-format",
        choices=["json", "text"],
        default="text",
        help="Report format (default: text)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        exit_code = asyncio.run(_run_cmd(args))
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
