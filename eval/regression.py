"""Regression case capture tool.

Captures a production miss as a benchmark case in eval/benchmark/fixtures/regression/
without modifying any application code.

Usage:
    python eval/regression.py add --case-file path/to/case.json
    python eval/regression.py add --help

Case file format (must match BenchmarkCase schema):
{
  "id": "regression-001",
  "verdict_class": "contradicted",
  "claim_text": "...",
  "evidence_spans": [{"span_text": "...", "relevance_score": 0.85, "rank": 1}],
  "gold_verdict": "contradicted",
  "source": "gold",
  "notes": "Production miss: model predicted supported, true answer is contradicted."
}
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on path when run as a script (python eval/regression.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.benchmark.schema import BenchmarkCase

REGRESSION_DIR = Path(__file__).parent / "benchmark" / "fixtures" / "regression"


def cmd_add(args: argparse.Namespace) -> int:
    case_file = Path(args.case_file)
    if not case_file.exists():
        print(f"ERROR: case file not found: {case_file}", file=sys.stderr)
        return 1

    try:
        raw = json.loads(case_file.read_text())
        case = BenchmarkCase.model_validate(raw)
    except Exception as exc:
        print(f"ERROR: case file is not valid BenchmarkCase: {exc}", file=sys.stderr)
        return 1

    REGRESSION_DIR.mkdir(parents=True, exist_ok=True)
    dest = REGRESSION_DIR / f"{case.id}.json"
    if dest.exists():
        print(f"WARNING: {dest} already exists — overwriting", file=sys.stderr)

    dest.write_text(json.dumps(raw, indent=2))
    print(f"Regression case written: {dest}")
    print(f"  id           : {case.id}")
    print(f"  verdict_class: {case.verdict_class}")
    print(f"  gold_verdict : {case.gold_verdict}")
    print(f"  source       : {case.source}")
    print()
    print("Next steps:")
    print("  1. git add eval/benchmark/fixtures/regression/")
    print(f"  2. git commit -m 'test(regression): add case {case.id}'")
    print("  No application code changes needed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regression",
        description="Capture a production miss as a regression test case",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_parser = subparsers.add_parser("add", help="Add a new regression case from a JSON file")
    add_parser.add_argument("--case-file", required=True, help="Path to a BenchmarkCase JSON file")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "add":
        sys.exit(cmd_add(args))


if __name__ == "__main__":
    main()
