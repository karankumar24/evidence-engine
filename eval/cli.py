"""Standalone CLI for the EvidenceEngine evaluation harness.

Requirements:
- OPENAI_API_KEY environment variable must be set for classify_claim to work
- No running FastAPI server is required — runner calls classify_claim() directly
- Runnable as: python -m eval.cli run [options]

Usage:
    python -m eval.cli run [--output-format {json|text}] [--model MODEL] [--concurrency N]
    python -m eval.cli run --help
"""

import argparse
import asyncio
import json
import sys

from eval.runner import run_evaluation


def _print_text_report(report, results) -> None:
    """Human-readable report to stdout."""
    print("=" * 60)
    print("EvidenceEngine Eval Report")
    print("=" * 60)
    gold_results = [r for r in results if r.source == "gold"]
    synth_results = [r for r in results if r.source == "synthetic"]
    print(f"Cases evaluated:  gold={len(gold_results)}  synthetic={len(synth_results)}")
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
    print("=" * 60)


def _print_json_report(report, results) -> None:
    """Machine-readable JSON report to stdout."""
    output = {
        "metrics": report.model_dump(),
        "summary": {
            "n_gold": len([r for r in results if r.source == "gold"]),
            "n_synthetic": len([r for r in results if r.source == "synthetic"]),
        },
    }
    print(json.dumps(output, indent=2))


async def _run_cmd(args: argparse.Namespace) -> int:
    """Execute the 'run' subcommand. Returns exit code."""
    report, results = await run_evaluation(
        model=args.model,
        concurrency=args.concurrency,
    )
    if args.output_format == "json":
        _print_json_report(report, results)
    else:
        _print_text_report(report, results)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval",
        description="EvidenceEngine evaluation harness — no web server required",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full benchmark evaluation")
    run_parser.add_argument(
        "--output-format",
        choices=["json", "text"],
        default="text",
        help="Output format for the eval report (default: text)",
    )
    run_parser.add_argument(
        "--model",
        default=None,
        help="Override the OpenAI model used for classification",
    )
    run_parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent classify_claim calls (default: 5)",
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
