"""Benchmark fixture loader — loads gold, synthetic, and regression cases."""

import json
from pathlib import Path

from eval.benchmark.schema import BenchmarkCase, BenchmarkSuite

FIXTURES_DIR = Path(__file__).parent / "fixtures"
VERDICT_CLASSES = ["supported", "contradicted", "insufficient_support", "needs_review"]
MIN_GOLD_PER_CLASS = 50


def load_benchmark() -> BenchmarkSuite:
    """Load all fixture files and return BenchmarkSuite with gold/synthetic separated.

    Loads from:
      - fixtures/gold/{verdict_class}.json       — arrays of gold cases
      - fixtures/synthetic/{verdict_class}.json  — arrays of synthetic cases
      - fixtures/regression/*.json               — individual gold regression cases

    Raises:
        ValueError: if any gold verdict class has fewer than MIN_GOLD_PER_CLASS cases.
    """
    gold_cases: list[BenchmarkCase] = []
    synthetic_cases: list[BenchmarkCase] = []

    for verdict_class in VERDICT_CLASSES:
        # Load gold
        gold_path = FIXTURES_DIR / "gold" / f"{verdict_class}.json"
        if gold_path.exists():
            raw = json.loads(gold_path.read_text())
            cases = [BenchmarkCase.model_validate(c) for c in raw]
            gold_cases.extend(c for c in cases if c.source == "gold")

        # Load synthetic
        synth_path = FIXTURES_DIR / "synthetic" / f"{verdict_class}.json"
        if synth_path.exists():
            raw = json.loads(synth_path.read_text())
            cases = [BenchmarkCase.model_validate(c) for c in raw]
            synthetic_cases.extend(c for c in cases if c.source == "synthetic")

    # Load regression cases (individual JSON objects, not arrays)
    regression_dir = FIXTURES_DIR / "regression"
    if regression_dir.exists():
        for reg_file in sorted(regression_dir.glob("*.json")):
            try:
                raw = json.loads(reg_file.read_text())
                case = BenchmarkCase.model_validate(raw)
                if case.source == "gold":
                    gold_cases.append(case)
                else:
                    synthetic_cases.append(case)
            except Exception:
                # Skip malformed regression files without crashing
                continue

    # Validate minimum gold coverage (EVAL-02)
    for verdict_class in VERDICT_CLASSES:
        class_gold = [c for c in gold_cases if c.gold_verdict == verdict_class]
        if len(class_gold) < MIN_GOLD_PER_CLASS:
            raise ValueError(
                f"Gold benchmark has only {len(class_gold)} '{verdict_class}' cases "
                f"(minimum required: {MIN_GOLD_PER_CLASS})"
            )

    return BenchmarkSuite(gold=gold_cases, synthetic=synthetic_cases)
