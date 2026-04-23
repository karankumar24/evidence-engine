"""CLI entrypoint for SciFact evaluation harness.

Usage:
    python -m eval.scifact run --split dev
    python -m eval.scifact run --split dev --resume
    python -m eval.scifact run --split dev --max-cases 10
    python -m eval.scifact run --split dev --max-cases 10 --concurrency 2

Subcommands:
    run     Run pipeline over claims split, write JSONL + metrics
    metrics Compute metrics from existing JSONL (no re-run)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys


def _cmd_run(args: argparse.Namespace) -> int:
    from eval.scifact.runner import run_scifact_evaluation  # noqa: PLC0415

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    result = asyncio.run(
        run_scifact_evaluation(
            split=args.split,
            resume=args.resume,
            max_cases=args.max_cases,
            concurrency=args.concurrency,
            results_dir=args.results_dir,
            cache_dir=args.cache_dir,
        )
    )

    print("\n=== SciFact Evaluation Results ===")
    print(f"macro_F1:       {result['macro_f1']:.4f}")
    print(f"support_F1:     {result['support']['f1']:.4f}  (n={result['support']['n']})")
    print(f"contradict_F1:  {result['contradict']['f1']:.4f}  (n={result['contradict']['n']})")
    print(f"n_nei:          {result['n_nei']}")
    print(f"n_total:        {result['n_total']}")
    if result.get("calibration", {}).get("ece") is not None:
        print(f"ECE:            {result['calibration']['ece']:.4f}")
        print(f"PNG:            {result['calibration'].get('output_png', 'N/A')}")
    print(f"Results JSONL:  {result['results_path']}")
    return 0


def _cmd_metrics(args: argparse.Namespace) -> int:
    """Compute metrics from an existing JSONL without re-running the pipeline."""
    from pathlib import Path
    from eval.scifact.metrics import compute_macro_f1  # noqa: PLC0415

    path = Path(args.results_file)
    if not path.exists():
        print(f"ERROR: results file not found: {path}", file=sys.stderr)
        return 1

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    result = compute_macro_f1(rows)
    print(json.dumps(result, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m eval.scifact",
        description="SciFact evaluation harness for EvidenceEngine",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run subcommand
    run_parser = subparsers.add_parser("run", help="Run evaluation pipeline")
    run_parser.add_argument("--split", choices=["dev", "train"], default="dev",
                            help="SciFact split to evaluate (default: dev = 300 claims)")
    run_parser.add_argument("--resume", action="store_true",
                            help="Skip claims already in results JSONL with status=classified")
    run_parser.add_argument("--max-cases", type=int, default=None, dest="max_cases",
                            help="Limit number of claims processed (smoke-test shortcut)")
    run_parser.add_argument("--concurrency", type=int, default=1,
                            help="Parallel classify calls (default 1 — safe for 8 GB M1 Air)")
    run_parser.add_argument("--results-dir", default="eval/results", dest="results_dir",
                            help="Output directory for JSONL results (default: eval/results)")
    run_parser.add_argument("--cache-dir", default="eval/data/scifact/", dest="cache_dir",
                            help="HuggingFace dataset cache directory")
    run_parser.set_defaults(func=_cmd_run)

    # metrics subcommand
    metrics_parser = subparsers.add_parser("metrics", help="Compute metrics from existing JSONL")
    metrics_parser.add_argument("results_file", help="Path to scifact-dev300.jsonl")
    metrics_parser.set_defaults(func=_cmd_metrics)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
